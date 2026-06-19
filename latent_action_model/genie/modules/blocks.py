import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor


def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, C, H, W  = videos.shape
    videos = videos[:, :, :, :H - (H % size), :W - (W % size)]
    x = rearrange(videos, "b t c (hn hp) (wn wp)  -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t c (hn hp) (wn wp) ", hp=size, wp=size, hn=hn)
    return x[:, :, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].cuda()


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim // 2)

    def scaled_dot_product_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False,
            attn_mask: Tensor = None,
    ) -> Tensor:
        scaled_attn_mask = None
        if attn_mask is not None:
            scaled_attn_mask = (attn_mask > 0).unsqueeze(1).unsqueeze(2)

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=scaled_attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

    def forward(self, x: Tensor, is_causal: bool = False, attn_mask: Tensor = None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)

        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False, attn_mask: Tensor = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_, is_causal=False, attn_mask=attn_mask)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False,
            to_out: bool = True,
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        if to_out:
            self.out = nn.Linear(model_dim, out_dim)
        else:
            self.out = nn.Identity()

        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, attn_mask)

        x = self.out(x)
        return x  # (B, T, E)


class MVSpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False,
            to_out: bool = True,
    ) -> None:
        super(MVSpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.view_embed = nn.Parameter(torch.zeros(2, model_dim), requires_grad=True)
        nn.init.normal_(self.view_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        if to_out:
            self.out = nn.Linear(model_dim, out_dim)
        else:
            self.out = nn.Identity()

        self.causal_temporal = causal_temporal

    def forward(self, latent_action: Tensor, view1: Tensor, view2: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        view1 = self.ffn(view1) + repeat(self.view_embed[0], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        view2 = self.ffn(view2) + repeat(self.view_embed[1], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        
        x = torch.cat([latent_action, view1, view2], dim=2)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, attn_mask)

        x = self.out(x)
        return x  # (B, T, E)

class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_, attn_mask=attn_mask)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.out(x)
        return x  # (B, T, E)


class MVSpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(MVSpatioTransformer, self).__init__()
        self.ffn = nn.Linear(in_dim, model_dim)

        self.pos_enc = PositionalEncoding(model_dim)
        # self.view_embed = nn.Parameter(torch.zeros(2, model_dim), requires_grad=True)
        # nn.init.normal_(self.view_embed, std=0.02)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, latent_action: Tensor, view1: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        view1 = self.ffn(view1) #+ repeat(self.view_embed[0], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        # view2 = self.ffn(view2) + repeat(self.view_embed[1], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        
        x = torch.cat([latent_action, view1], dim=2)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.out(x)
        return x  # (B, T, E)


class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = True) -> None:
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.num_latents = num_latents
        self.codebook_init_mode = os.environ.get("POLAR_VQ_INIT_MODE", "uniform").strip().lower()
        self.codebook_init_range = float(os.environ.get("POLAR_VQ_INIT_RANGE", str(1.0 / num_latents)))
        if self.codebook_init_range <= 0.0:
            raise ValueError(f"POLAR_VQ_INIT_RANGE must be > 0, got {self.codebook_init_range}.")
        self._init_codebook_weight()

        # Initialize a usage buffer
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.register_buffer("_restart_count", torch.zeros((), dtype=torch.long), persistent=False)

        self.code_restart = code_restart

    def _init_generator(self) -> torch.Generator:
        seed = int(os.environ.get("POLAR_VQ_INIT_SEED", "1729"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _parse_csv_floats(value: str, name: str) -> Tuple[float, ...]:
        try:
            parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated float list, got {value!r}.") from exc
        if not parsed:
            raise ValueError(f"{name} must not be empty.")
        return parsed

    @staticmethod
    def _parse_csv_ints(value: str, name: str) -> Tuple[int, ...]:
        try:
            parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated int list, got {value!r}.") from exc
        if not parsed:
            raise ValueError(f"{name} must not be empty.")
        if any(count <= 0 for count in parsed):
            raise ValueError(f"{name} entries must be positive, got {value!r}.")
        return parsed

    def _zero_shell_init_weight(self) -> Tensor:
        radii = self._parse_csv_floats(
            os.environ.get("POLAR_VQ_INIT_SHELL_RADII", "1.0,2.5,4.0"),
            "POLAR_VQ_INIT_SHELL_RADII",
        )
        counts = self._parse_csv_ints(
            os.environ.get("POLAR_VQ_INIT_SHELL_COUNTS", "5,5,5"),
            "POLAR_VQ_INIT_SHELL_COUNTS",
        )
        if len(radii) != len(counts):
            raise ValueError(
                "POLAR_VQ_INIT_SHELL_RADII and POLAR_VQ_INIT_SHELL_COUNTS "
                f"must have the same length, got {len(radii)} and {len(counts)}."
            )
        if sum(counts) != self.num_latents - 1:
            raise ValueError(
                "POLAR_VQ_INIT_SHELL_COUNTS must sum to num_latents - 1 "
                f"({self.num_latents - 1}), got {sum(counts)}."
            )

        generator = self._init_generator()
        weight = torch.zeros(self.num_latents, self.codebook.embedding_dim, dtype=torch.float32)
        cursor = 1
        for radius, count in zip(radii, counts):
            if radius <= 0.0:
                raise ValueError(f"POLAR_VQ_INIT_SHELL_RADII entries must be > 0, got {radius}.")
            direction = torch.randn(
                count,
                self.codebook.embedding_dim,
                generator=generator,
                dtype=torch.float32,
            )
            direction = F.normalize(direction, dim=-1, eps=1e-6)
            weight[cursor : cursor + count] = direction * radius
            cursor += count
        return weight

    def _init_codebook_weight(self) -> None:
        if self.codebook_init_mode == "uniform":
            self.codebook.weight.data.uniform_(-self.codebook_init_range, self.codebook_init_range)
            return
        if self.codebook_init_mode == "zero_shells":
            weight = self._zero_shell_init_weight().to(
                device=self.codebook.weight.device,
                dtype=self.codebook.weight.dtype,
            )
            self.codebook.weight.data.copy_(weight)
            return
        raise ValueError(
            "POLAR_VQ_INIT_MODE must be one of {'uniform', 'zero_shells'}, "
            f"got {self.codebook_init_mode!r}."
        )

    def _global_usage(self) -> Tensor:
        usage = self.usage.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(usage, op=torch.distributed.ReduceOp.SUM)
        return usage

    def _restart_generator(self) -> torch.Generator:
        seed = int(os.environ.get("POLAR_VQ_RESTART_SEED", "1729")) + int(self._restart_count.item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _rank0_print(self, message: str) -> None:
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank == 0:
            print(message, flush=True)

    def update_usage(self, min_enc) -> None:
        counts = torch.bincount(
            min_enc.detach().reshape(-1),
            minlength=self.num_latents,
        ).to(self.usage)
        self.usage.add_(counts)

    def random_restart(self) -> None:
        if self.code_restart:
            # Randomly restart all dead codes
            usage = self._global_usage()
            dead_codes = torch.nonzero(usage < 1).squeeze(1)
            dead_count = dead_codes.numel()
            live_codes = torch.nonzero(usage >= 1).squeeze(1)
            self._rank0_print(f"Restarting {dead_count} codes")
            with torch.no_grad():
                if dead_count > 0:
                    generator = self._restart_generator()
                    if live_codes.numel() > 0:
                        repeat_count = math.ceil(dead_count / live_codes.numel())
                        source_codes = live_codes.repeat(repeat_count)[:dead_count]
                        restarted = self.codebook.weight[source_codes].detach().clone()
                        noise_std = float(os.environ.get("POLAR_VQ_RESTART_NOISE", "0.01"))
                        if noise_std > 0.0:
                            noise = torch.randn(
                                restarted.shape,
                                generator=generator,
                                dtype=torch.float32,
                            ).to(device=restarted.device, dtype=restarted.dtype)
                            restarted = restarted + noise_std * noise
                    else:
                        restarted = torch.empty(
                            dead_count,
                            self.codebook.embedding_dim,
                            dtype=torch.float32,
                        )
                        restarted.uniform_(
                            -self.codebook_init_range,
                            self.codebook_init_range,
                            generator=generator,
                        )
                        restarted = restarted.to(
                            device=self.codebook.weight.device,
                            dtype=self.codebook.weight.dtype,
                        )
                    self.codebook.weight[dead_codes] = restarted
                    self._restart_count.add_(1)

            if hasattr(self, "inner_vq"):
                self.inner_vq.random_restart()

    def reset_usage(self) -> None:
        if self.code_restart:
            # Reset usage between epochs
            self.usage.zero_()

            if hasattr(self, "inner_vq"):
                self.inner_vq.reset_usage()

    def forward(self, x: Tensor, update_usage: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=-1)
        # indices = torch.randint(0, 31, (8,4)).to('cuda')
        z = self.codebook(indices)
        
        # Update code usage
        if (
            update_usage
            and (not self.training or self.code_restart)
            and os.environ.get("POLAR_LAM_DISABLE_USAGE_UPDATE", "0") != "1"
        ):
            self.update_usage(indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q, z, x, indices


class FactorizedVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_radius_codes: int,
        num_direction_codes: int,
        latent_dim: int,
        radius_values: str | None = None,
        code_restart: bool = True,
    ) -> None:
        super().__init__()
        if num_radius_codes < 2:
            raise ValueError(f"num_radius_codes must be >= 2, got {num_radius_codes}.")
        if num_direction_codes < 1:
            raise ValueError(f"num_direction_codes must be >= 1, got {num_direction_codes}.")

        self.num_radius_codes = int(num_radius_codes)
        self.num_direction_codes = int(num_direction_codes)
        # Token vocabulary size: one radius token plus one direction token per slot.
        # The geometric codebook still has R * D possible vectors via rho[r] * dir[d].
        self.num_latents = self.num_radius_codes + self.num_direction_codes
        self.latent_dim = int(latent_dim)
        self.code_restart = code_restart
        self.radius_delta_floor = float(os.environ.get("POLAR_FACTOR_VQ_RADIUS_DELTA_FLOOR", "1e-5"))
        if self.radius_delta_floor <= 0.0:
            raise ValueError(
                "POLAR_FACTOR_VQ_RADIUS_DELTA_FLOOR must be > 0, "
                f"got {self.radius_delta_floor}."
            )

        radii = self._init_radius_values(radius_values)
        self.radius_has_zero = bool(radii[0].item() <= self.radius_delta_floor)
        if self.radius_has_zero:
            deltas = radii[1:] - radii[:-1]
        else:
            deltas = torch.cat([radii[:1], radii[1:] - radii[:-1]], dim=0)
        if torch.any(deltas <= 0):
            raise ValueError(f"factorized VQ radius values must be strictly increasing, got {radii.tolist()}.")
        deltas = torch.clamp(deltas - self.radius_delta_floor, min=self.radius_delta_floor)
        self.radius_delta_unconstrained = nn.Parameter(self._inverse_softplus(deltas))

        self.direction_codebook = nn.Embedding(self.num_direction_codes, self.latent_dim)
        self._init_direction_weight()

        self.register_buffer("usage", torch.zeros(self.num_latents), persistent=False)
        self.register_buffer("radius_usage", torch.zeros(self.num_radius_codes), persistent=False)
        self.register_buffer("direction_usage", torch.zeros(self.num_direction_codes), persistent=False)
        self.register_buffer("_restart_count", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return torch.log(torch.expm1(value))

    def _init_radius_values(self, radius_values: str | None) -> Tensor:
        value = radius_values or os.environ.get("POLAR_FACTOR_VQ_RADIUS_VALUES", "")
        if value:
            radii = torch.tensor(VectorQuantizer._parse_csv_floats(value, "factorized_vq_radius_values"))
        else:
            radii = torch.linspace(0.0, 1.0, self.num_radius_codes)
        if radii.numel() != self.num_radius_codes:
            raise ValueError(
                "factorized_vq_radius_values must contain exactly "
                f"{self.num_radius_codes} values, got {radii.numel()}."
            )
        if torch.any(radii < 0):
            raise ValueError(f"factorized VQ radius values must be >= 0, got {radii.tolist()}.")
        if torch.any(radii[1:] <= radii[:-1]):
            raise ValueError(f"factorized VQ radius values must be strictly increasing, got {radii.tolist()}.")
        return radii.float()

    def _init_generator(self) -> torch.Generator:
        seed = int(os.environ.get("POLAR_FACTOR_VQ_INIT_SEED", "1729"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _init_direction_weight(self) -> None:
        generator = self._init_generator()
        init_range = float(os.environ.get("POLAR_FACTOR_VQ_DIRECTION_INIT_RANGE", str(1.0 / self.num_latents)))
        if init_range <= 0.0:
            raise ValueError(
                "POLAR_FACTOR_VQ_DIRECTION_INIT_RANGE must be > 0, "
                f"got {init_range}."
            )
        weight = torch.empty(
            self.num_direction_codes,
            self.latent_dim,
            dtype=torch.float32,
        )
        weight.uniform_(-init_range, init_range, generator=generator)
        weight = F.normalize(weight, dim=-1, eps=1e-6)
        self.direction_codebook.weight.data.copy_(weight.to(self.direction_codebook.weight))

    def radius_values(self) -> Tensor:
        deltas = F.softplus(self.radius_delta_unconstrained) + self.radius_delta_floor
        radii = torch.cumsum(deltas, dim=0)
        if self.radius_has_zero:
            radii = torch.cat([radii.new_zeros(1), radii], dim=0)
        return radii

    def direction_values(self) -> Tensor:
        return F.normalize(self.direction_codebook.weight, dim=-1, eps=1e-6)

    def codebook_weight(self) -> Tensor:
        radii = self.radius_values()
        directions = self.direction_values()
        return (radii[:, None, None] * directions[None, :, :]).reshape(
            self.num_radius_codes * self.num_direction_codes,
            self.latent_dim,
        )

    def update_usage(self, radius_indices: Tensor, direction_indices: Tensor, token_indices: Tensor) -> None:
        self.usage.add_(torch.bincount(token_indices.detach().reshape(-1), minlength=self.num_latents).to(self.usage))
        self.radius_usage.add_(
            torch.bincount(radius_indices.detach().reshape(-1), minlength=self.num_radius_codes).to(self.radius_usage)
        )
        self.direction_usage.add_(
            torch.bincount(
                direction_indices.detach().reshape(-1),
                minlength=self.num_direction_codes,
            ).to(self.direction_usage)
        )

    def _global_usages(self) -> Tuple[Tensor, Tensor, Tensor]:
        usage = self.usage.clone()
        radius_usage = self.radius_usage.clone()
        direction_usage = self.direction_usage.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(usage, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(radius_usage, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(direction_usage, op=torch.distributed.ReduceOp.SUM)
        return usage, radius_usage, direction_usage

    def _restart_generator(self) -> torch.Generator:
        seed = int(os.environ.get("POLAR_FACTOR_VQ_RESTART_SEED", "1729")) + int(self._restart_count.item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def random_restart(self) -> None:
        if not self.code_restart:
            return
        usage, radius_usage, direction_usage = self._global_usages()
        dead_tokens = torch.nonzero(usage < 1).squeeze(1)
        dead_radii = torch.nonzero(radius_usage < 1).squeeze(1)
        dead_directions = torch.nonzero(direction_usage < 1).squeeze(1)
        live_directions = torch.nonzero(direction_usage >= 1).squeeze(1)

        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        if rank == 0:
            print(
                "FactorizedVQ usage: "
                f"dead_tokens={dead_tokens.numel()} "
                f"dead_radius={dead_radii.numel()} "
                f"dead_direction={dead_directions.numel()}",
                flush=True,
            )
        if dead_directions.numel() == 0:
            return

        with torch.no_grad():
            generator = self._restart_generator()
            if live_directions.numel() > 0:
                repeat_count = math.ceil(dead_directions.numel() / live_directions.numel())
                source_directions = live_directions.repeat(repeat_count)[: dead_directions.numel()]
                restarted = self.direction_codebook.weight[source_directions].detach().clone()
                noise_std = float(os.environ.get("POLAR_FACTOR_VQ_RESTART_NOISE", "0.01"))
                if noise_std > 0.0:
                    noise = torch.randn(
                        restarted.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ).to(device=restarted.device, dtype=restarted.dtype)
                    restarted = restarted + noise_std * noise
            else:
                init_range = float(os.environ.get("POLAR_FACTOR_VQ_DIRECTION_INIT_RANGE", str(1.0 / self.num_latents)))
                restarted = torch.empty(
                    dead_directions.numel(),
                    self.latent_dim,
                    dtype=torch.float32,
                )
                restarted.uniform_(-init_range, init_range, generator=generator)
                restarted = restarted.to(
                    device=self.direction_codebook.weight.device,
                    dtype=self.direction_codebook.weight.dtype,
                )
            restarted = F.normalize(restarted, dim=-1, eps=1e-6)
            self.direction_codebook.weight[dead_directions] = restarted
            self._restart_count.add_(1)

    def reset_usage(self) -> None:
        if self.code_restart:
            self.usage.zero_()
            self.radius_usage.zero_()
            self.direction_usage.zero_()

    def forward(self, x: Tensor, update_usage: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        radii = self.radius_values()
        directions = self.direction_values()

        radius = torch.linalg.vector_norm(x, dim=-1).mean(dim=-1)
        unit = F.normalize(x, dim=-1, eps=1e-6)
        radius_indices = torch.argmin((radius[..., None] - radii) ** 2, dim=-1)
        direction_indices = torch.argmax(unit @ directions.T, dim=-1)
        direction_token_indices = direction_indices + self.num_radius_codes
        token_indices = torch.cat([radius_indices[..., None], direction_token_indices], dim=-1)

        z = radii[radius_indices][..., None, None] * directions[direction_indices]
        if (
            update_usage
            and (not self.training or self.code_restart)
            and os.environ.get("POLAR_LAM_DISABLE_USAGE_UPDATE", "0") != "1"
        ):
            self.update_usage(radius_indices, direction_indices, token_indices)

        z_q = x + (z - x).detach()
        return z_q, z, x, token_indices, radius_indices, direction_indices


class ResidualVectorQuantizer(VectorQuantizer):
    def __init__(self, num_latents: int, latent_dim: int) -> None:
        super(ResidualVectorQuantizer, self).__init__(num_latents, latent_dim)
        self.inner_vq = VectorQuantizer(num_latents, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Compute distances
        distance = torch.cdist(x, self.codebook.weight)

        # Get indices and embeddings
        indices = torch.argmin(distance, dim=1)

        z = self.codebook(indices)

        # Residual quantization
        residual = x - z.detach()
        inner_z_q, inner_z, inner_x, inner_indices = self.inner_vq(residual)

        # Update code usage
        if not self.training or self.code_restart:
            self.update_usage(indices)
            self.inner_vq.update_usage(inner_indices)

        # Straight through estimator
        z_q = x + (z - x).detach()
        return z_q + inner_z_q, z, x, indices, inner_z, inner_x, inner_indices
