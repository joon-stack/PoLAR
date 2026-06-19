import os
import time
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchvision import transforms

from latent_action_model.genie.modules.blocks import (
    FactorizedVectorQuantizer,
    SpatioTemporalTransformer,
    SpatioTransformer,
    VectorQuantizer,
)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VisualVQDINOLatentActionModel(nn.Module):
    """
    Visual-only latent action VQ-VAE operating in DINO patch-token space.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        num_codes: int = 4,
        dropout: float = 0.0,
        hyperbolic_action_prelift_enabled: bool = False,
        hyperbolic_curvature: float = 1.0,
        hyperbolic_prelift_mode: str = "none",
        hyperbolic_prelift_scale: float = 1.0,
        hyperbolic_tangent_max_norm: float = 0.0,
        hyperbolic_lift_max_norm: float = 0.0,
        hyperbolic_eps: float = 1e-5,
        latent_output_global_scale: float = 1.0,
        use_vq: bool = True,
        vq_type: str = "standard",
        factorized_vq_num_radius: int = 4,
        factorized_vq_num_directions: int = 4,
        factorized_vq_radius_values: str | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_codes = int(num_codes)
        if self.num_codes <= 0:
            raise ValueError(f"num_codes must be > 0, got {self.num_codes}.")
        self.use_vq = bool(use_vq)
        self.vq_type = str(vq_type).strip().lower()
        if self.vq_type not in {"standard", "factorized"}:
            raise ValueError(f"vq_type must be one of {{'standard', 'factorized'}}, got {self.vq_type!r}.")
        self.factorized_vq_num_radius = int(factorized_vq_num_radius)
        self.factorized_vq_num_directions = int(factorized_vq_num_directions)
        self.hyperbolic_action_prelift_enabled = hyperbolic_action_prelift_enabled
        self.hyperbolic_curvature = float(hyperbolic_curvature)
        self.hyperbolic_prelift_mode = str(hyperbolic_prelift_mode).strip().lower()
        self.hyperbolic_prelift_scale = float(hyperbolic_prelift_scale)
        self.hyperbolic_tangent_max_norm = float(hyperbolic_tangent_max_norm)
        self.hyperbolic_lift_max_norm = float(hyperbolic_lift_max_norm)
        self.hyperbolic_eps = float(hyperbolic_eps)
        self.latent_output_global_scale = float(latent_output_global_scale)
        self._validate_hyperbolic_action_config()

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        if self.vq_type == "factorized":
            self.vq = FactorizedVectorQuantizer(
                num_radius_codes=self.factorized_vq_num_radius,
                num_direction_codes=self.factorized_vq_num_directions,
                latent_dim=latent_dim,
                radius_values=factorized_vq_radius_values,
                code_restart=True,
            )
            if self.vq.num_latents != num_latents:
                raise ValueError(
                    "For factorized VQ, lam_num_latents must equal "
                    "factorized_vq_num_radius + factorized_vq_num_directions "
                    f"({self.vq.num_latents}), got {num_latents}."
                )
        else:
            self.vq = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        if not self.use_vq:
            self.vq.requires_grad_(False)

        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self._module_profile_steps = int(os.environ.get("POLAR_LAM_MODULE_PROFILE_STEPS", "0"))
        self._module_profile_count = 0

    def _validate_hyperbolic_action_config(self) -> None:
        if self.hyperbolic_curvature <= 0.0:
            raise ValueError(f"hyperbolic_curvature must be > 0, got {self.hyperbolic_curvature}.")
        if self.hyperbolic_prelift_mode not in {"none", "scale"}:
            raise ValueError(
                "hyperbolic_prelift_mode must be one of {'none', 'scale'}, "
                f"got {self.hyperbolic_prelift_mode!r}."
            )
        if self.hyperbolic_prelift_scale <= 0.0:
            raise ValueError(f"hyperbolic_prelift_scale must be > 0, got {self.hyperbolic_prelift_scale}.")
        if self.hyperbolic_tangent_max_norm < 0.0:
            raise ValueError(
                f"hyperbolic_tangent_max_norm must be >= 0, got {self.hyperbolic_tangent_max_norm}."
            )
        if self.hyperbolic_lift_max_norm < 0.0:
            raise ValueError(f"hyperbolic_lift_max_norm must be >= 0, got {self.hyperbolic_lift_max_norm}.")
        if self.hyperbolic_eps <= 0.0:
            raise ValueError(f"hyperbolic_eps must be > 0, got {self.hyperbolic_eps}.")
        if self.latent_output_global_scale <= 0.0:
            raise ValueError(
                f"latent_output_global_scale must be > 0, got {self.latent_output_global_scale}."
            )

    def _clamp_norm(self, tensor: Tensor, max_norm: float) -> Tensor:
        if max_norm <= 0.0:
            return tensor
        norm = tensor.norm(dim=-1, keepdim=True).clamp_min(self.hyperbolic_eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return tensor * scale

    def _hyperbolic_prelift_transform(self, tangent: Tensor) -> Tensor:
        tangent = tangent * self.latent_output_global_scale
        if self.hyperbolic_prelift_mode == "scale":
            tangent = tangent * self.hyperbolic_prelift_scale
        return self._clamp_norm(tangent, self.hyperbolic_tangent_max_norm)

    def _prelift_latents(self, latents: Tensor) -> Tensor:
        if not self.hyperbolic_action_prelift_enabled:
            return latents
        return self._hyperbolic_prelift_transform(latents)

    def _extract_dino_features(self, videos: Tensor) -> Tensor:
        bsz, num_frames = videos.shape[:2]
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)["x_norm_patchtokens"]
        return rearrange(dino_features, "(b t) l d -> b t l d", b=bsz, t=num_frames)

    def _encode_transition_features(self, current_features: Tensor, target_features: Tensor) -> Tensor:
        bsz = current_features.shape[0]
        features = torch.stack([current_features, target_features], dim=1)
        action_pad = self.action_latent.expand(bsz, 2, -1, -1)
        padded_patches = torch.cat([action_pad, features], dim=2)
        z = self.encoder(padded_patches)
        return self._prelift_latents(self.to_codebook(z[:, 1, :self.num_codes]))

    def encode_transition_tokens(self, current_images: Tensor, target_images: Tensor) -> Tensor:
        videos = torch.stack([current_images, target_images], dim=1)
        features = self._extract_dino_features(videos)
        return self._encode_transition_features(features[:, 0], features[:, 1])

    def _run_vq(self, latents: Tensor, update_usage: bool = True) -> Dict[str, Tensor]:
        vq_outputs = self.vq(latents, update_usage=update_usage)
        z_q, z, emb, indices = vq_outputs[:4]
        outputs = {
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }
        if len(vq_outputs) > 4:
            outputs["radius_indices"] = vq_outputs[4]
            outputs["direction_indices"] = vq_outputs[5]
        return outputs

    def vq_encode(self, videos: Tensor) -> Dict:
        profile = (
            self._module_profile_steps
            and self._module_profile_count < self._module_profile_steps
            and os.environ.get("LOCAL_RANK", "0") == "0"
        )
        times = {}

        def mark(name: str, start: float) -> float:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if profile:
                times[name] = now - start
            return now

        start = time.perf_counter()
        bsz, num_frames = videos.shape[:2]
        dino_features = self._extract_dino_features(videos)
        start = mark("dino", start)

        action_pad = self.action_latent.expand(bsz, num_frames, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        z = self.encoder(padded_patches)
        start = mark("encoder", start)

        emb_raw = self.to_codebook(z[:, 1:, :self.num_codes])
        emb_raw = emb_raw.reshape(bsz * (num_frames - 1), self.num_codes, self.latent_dim)
        emb_in = self._prelift_latents(emb_raw)
        if self.use_vq:
            vq_outputs = self._run_vq(emb_in)
        else:
            vq_outputs = {
                "z_q": emb_in,
                "z": emb_in.detach(),
                "emb": emb_in,
                "indices": torch.zeros(emb_in.shape[:-1], dtype=torch.long, device=emb_in.device),
            }
        z_q = vq_outputs["z_q"]
        z_q = z_q.reshape(bsz, num_frames - 1, self.num_codes, self.latent_dim)
        mark("vq", start)

        outputs = {
            "patches": dino_features,
            "z_q": z_q,
            "z": vq_outputs["z"],
            "emb": vq_outputs["emb"],
            "emb_raw": emb_raw,
            "indices": vq_outputs["indices"],
            "continuous_latent": z_q,
            "use_vq": self.use_vq,
            "_profile_times": times,
        }
        if "radius_indices" in vq_outputs:
            outputs["radius_indices"] = vq_outputs["radius_indices"]
            outputs["direction_indices"] = vq_outputs["direction_indices"]
        return outputs

    def forward(self, batch: Dict) -> Dict:
        profile = (
            self._module_profile_steps
            and self._module_profile_count < self._module_profile_steps
            and os.environ.get("LOCAL_RANK", "0") == "0"
        )

        outputs = self.vq_encode(batch["videos"])
        source_features = outputs["patches"][:, :-1]
        target_features = outputs["patches"][:, 1:]

        if "radprog_mid_pixel_values" in batch:
            current_features = outputs["patches"][:, 0]
            future_features = outputs["patches"][:, -1]
            mid_features = self._extract_dino_features(batch["radprog_mid_pixel_values"].unsqueeze(1))[:, 0]

            z_future = outputs["emb"]
            z_future_code = outputs["z"]
            z_future_q = outputs["z_q"][:, -1]
            z_mid = self._encode_transition_features(current_features, mid_features)
            z_self = self._encode_transition_features(current_features, current_features)
            if self.use_vq:
                z_mid_outputs = self._run_vq(z_mid)
                z_self_outputs = self._run_vq(z_self, update_usage=False)
                z_mid_q = z_mid_outputs["z_q"]
                z_mid_code = z_mid_outputs["z"]
                z_mid_emb = z_mid_outputs["emb"]
                z_mid_indices = z_mid_outputs["indices"]
                z_self_q = z_self_outputs["z_q"]
                z_self_code = z_self_outputs["z"]
            else:
                z_mid_q = z_mid
                z_mid_code = z_mid.detach()
                z_mid_emb = z_mid
                z_mid_indices = torch.zeros(z_mid.shape[:-1], dtype=torch.long, device=z_mid.device)
                z_self_q = z_self
                z_self_code = z_self.detach()

            outputs["z_q"] = torch.cat([z_mid_q[:, None], outputs["z_q"]], dim=1)
            outputs["z"] = torch.cat([z_mid_code, outputs["z"]], dim=0)
            outputs["emb"] = torch.cat([z_mid_emb, outputs["emb"]], dim=0)
            outputs["indices"] = torch.cat([z_mid_indices, outputs["indices"]], dim=0)
            if self.use_vq and "radius_indices" in z_mid_outputs and "radius_indices" in outputs:
                outputs["radius_indices"] = torch.cat(
                    [z_mid_outputs["radius_indices"], outputs["radius_indices"]],
                    dim=0,
                )
                outputs["direction_indices"] = torch.cat(
                    [z_mid_outputs["direction_indices"], outputs["direction_indices"]],
                    dim=0,
                )

            outputs["radprog_z_self"] = z_self
            outputs["radprog_z_mid"] = z_mid_emb
            outputs["radprog_z_future"] = z_future
            outputs["radprog_zq_self"] = z_self_q
            outputs["radprog_zq_mid"] = z_mid_q
            outputs["radprog_zq_future"] = z_future_q
            if self.use_vq:
                outputs["radprog_code_self"] = z_self_code
                outputs["radprog_code_mid"] = z_mid_code
                outputs["radprog_code_future"] = z_future_code

            source_features = current_features[:, None].expand(-1, 2, -1, -1)
            target_features = torch.stack([mid_features, future_features], dim=1)

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        decoder_start = time.perf_counter()

        video_patches = self.patch_up(source_features)
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes : self.num_codes + video_patches.shape[2]]

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = outputs.pop("_profile_times")
            times["decoder"] = time.perf_counter() - decoder_start
            print(
                "LAM_MODULE_PROFILE "
                f"step={self._module_profile_count + 1:06d} "
                f"dino={times.get('dino', 0.0):.3f}s "
                f"encoder={times.get('encoder', 0.0):.3f}s "
                f"vq={times.get('vq', 0.0):.3f}s "
                f"decoder={times.get('decoder', 0.0):.3f}s",
                flush=True,
            )
        else:
            outputs.pop("_profile_times", None)
        self._module_profile_count += 1

        outputs.update(
            {
                "recon": video_recon,
                "target": target_features,
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
