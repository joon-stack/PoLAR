import os
import time
from os import listdir, makedirs
from typing import Any, Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from accelerate import PartialState
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from genie.modules import VisualVQDINOLatentActionModel


OptimizerCallable = Callable[[Iterable], Optimizer]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


class VisualVQ_DINO_LAM(LightningModule):
    """
    Single-stage visual-only latent action model operating at DINO latent space.
    """

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_num_latents: int = 8,
        lam_num_codes: int = 4,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        use_vq: bool = True,
        vq_type: str = "standard",
        factorized_vq_num_radius: int = 4,
        factorized_vq_num_directions: int = 4,
        factorized_vq_radius_values: str | None = None,
        vq_beta: float = 0.25,
        vq_restart_interval_steps: int = 0,
        radprog_radial_weight: float = 0.0,
        radprog_progress_weight: float = 0.0,
        radprog_codebook_aux_weight: float = 0.0,
        radprog_warmup_steps: int = 0,
        radprog_progress_alpha: float = 0.05,
        hyperbolic_latent_enabled: bool = False,
        hyperbolic_curvature: float = 1.0,
        hyperbolic_prelift_mode: str = "none",
        hyperbolic_prelift_scale: float = 1.0,
        hyperbolic_tangent_max_norm: float = 0.0,
        hyperbolic_lift_max_norm: float = 0.0,
        hyperbolic_eps: float = 1e-5,
        latent_output_global_scale: float = 1.0,
        action_probe_enabled: bool = False,
        action_probe_shuffle: bool = True,
        action_probe_lr: float = 1e-3,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        task_name: str = "visual_vq_lam_bridge",
        optimizer: OptimizerCallable = AdamW,
        make_data_pair: bool = False,
    ) -> None:
        super().__init__()

        self.lam = VisualVQDINOLatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            num_latents=lam_num_latents,
            num_codes=lam_num_codes,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            hyperbolic_action_prelift_enabled=hyperbolic_latent_enabled,
            hyperbolic_curvature=hyperbolic_curvature,
            hyperbolic_prelift_mode=hyperbolic_prelift_mode,
            hyperbolic_prelift_scale=hyperbolic_prelift_scale,
            hyperbolic_tangent_max_norm=hyperbolic_tangent_max_norm,
            hyperbolic_lift_max_norm=hyperbolic_lift_max_norm,
            hyperbolic_eps=hyperbolic_eps,
            latent_output_global_scale=latent_output_global_scale,
            use_vq=use_vq,
            vq_type=vq_type,
            factorized_vq_num_radius=factorized_vq_num_radius,
            factorized_vq_num_directions=factorized_vq_num_directions,
            factorized_vq_radius_values=factorized_vq_radius_values,
        )

        self.use_vq = bool(use_vq)
        self.vq_type = str(vq_type).strip().lower()
        self.factorized_vq_num_radius = int(factorized_vq_num_radius)
        self.factorized_vq_num_directions = int(factorized_vq_num_directions)
        self.lam_num_codes = int(lam_num_codes)
        self.lam_num_latents = self.lam.vq.num_latents if self.use_vq else lam_num_latents
        self.vq_beta = vq_beta
        self.vq_restart_interval_steps = int(vq_restart_interval_steps)
        if self.vq_restart_interval_steps < 0:
            raise ValueError(
                f"vq_restart_interval_steps must be >= 0, got {self.vq_restart_interval_steps}."
            )
        self.radprog_radial_weight = radprog_radial_weight
        self.radprog_progress_weight = radprog_progress_weight
        self.radprog_codebook_aux_weight = float(radprog_codebook_aux_weight)
        if self.radprog_codebook_aux_weight < 0.0:
            raise ValueError(
                f"radprog_codebook_aux_weight must be >= 0, got {self.radprog_codebook_aux_weight}."
            )
        self.radprog_warmup_steps = int(radprog_warmup_steps)
        if self.radprog_warmup_steps < 0:
            raise ValueError(f"radprog_warmup_steps must be >= 0, got {self.radprog_warmup_steps}.")
        self.radprog_progress_alpha = radprog_progress_alpha
        self.hyperbolic_latent_enabled = hyperbolic_latent_enabled
        self.hyperbolic_curvature = float(hyperbolic_curvature)
        self.hyperbolic_prelift_mode = str(hyperbolic_prelift_mode).strip().lower()
        self.hyperbolic_prelift_scale = float(hyperbolic_prelift_scale)
        self.hyperbolic_tangent_max_norm = float(hyperbolic_tangent_max_norm)
        self.hyperbolic_lift_max_norm = float(hyperbolic_lift_max_norm)
        self.hyperbolic_eps = float(hyperbolic_eps)
        self.latent_output_global_scale = float(latent_output_global_scale)
        self._validate_hyperbolic_config()
        self.action_probe_enabled = action_probe_enabled
        self.action_probe_shuffle = action_probe_shuffle
        self.action_probe_lr = action_probe_lr
        # Probe weights are diagnostics only; keep them out of Lightning state_dict/checkpoints.
        self._action_probe_state = {
            "probe": None,
            "optimizer": None,
            "shuffled": None,
            "shuffled_optimizer": None,
        }
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        self._profile_steps = int(os.environ.get("POLAR_LAM_PROFILE_STEPS", "0"))
        self._profile_data_wait = 0.0
        self._profile_batch_start = None
        self._profile_shared_step = 0.0
        self._profile_backward_start = None
        self._profile_backward = 0.0
        self._profile_prev_batch_end = None
        self._profile_active_batch = False
        self._profile_batch_transfer = 0.0
        self._profile_batch_transfer_start = None
        self._profile_input: Dict[str, float] = {}
        self._use_wandb = _env_flag("POLAR_LAM_USE_WANDB") and self.distributed_state.is_main_process
        if self._use_wandb:
            wandb.init(name=task_name, reinit=True)

    def _code_usage(self, indices: Tensor, num_latents: int) -> Tensor:
        counts = torch.bincount(
            indices.detach().reshape(-1),
            minlength=num_latents,
        )
        return (counts != 0).float().mean()

    def _entropy_from_counts(self, counts: Tensor) -> Tensor:
        counts = counts.detach().float()
        total = counts.sum()
        if total <= 0:
            return counts.new_zeros(())
        probs = counts / total
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum()

    def _token_entropy(self, indices: Tensor, num_latents: int) -> Tensor:
        counts = torch.bincount(
            indices.detach().reshape(-1).long(),
            minlength=num_latents,
        )
        return self._entropy_from_counts(counts)

    def _token_perplexity(self, indices: Tensor, num_latents: int) -> Tensor:
        return self._token_entropy(indices, num_latents).exp()

    def _radius_direction_pair_indices(self, radius_indices: Tensor, direction_indices: Tensor) -> Tensor:
        radius_flat = radius_indices.detach().reshape(-1, 1).long()
        direction_flat = direction_indices.detach().reshape(radius_flat.shape[0], -1).long()
        return (radius_flat * self.factorized_vq_num_directions + direction_flat).reshape(-1)

    def _radius_direction_pair_usage(self, radius_indices: Tensor, direction_indices: Tensor) -> Tensor:
        pair_indices = self._radius_direction_pair_indices(radius_indices, direction_indices)
        return self._code_usage(
            pair_indices,
            self.factorized_vq_num_radius * self.factorized_vq_num_directions,
        )

    def _rank_bins(self, values: Tensor, num_bins: int) -> Tuple[Tensor, int]:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return values.new_zeros((0,), dtype=torch.long), 1
        if values.numel() <= 1 or values.var(unbiased=False) <= 1e-12:
            return torch.zeros_like(values, dtype=torch.long), 1
        bins = max(1, min(int(num_bins), int(values.numel())))
        order = torch.argsort(values)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(values.numel(), device=values.device, dtype=order.dtype)
        return torch.div(ranks * bins, values.numel(), rounding_mode="floor").long().clamp_max(bins - 1), bins

    def _discrete_mi_nmi(self, x: Tensor, num_x: int, y: Tensor, num_y: int) -> Tuple[Tensor, Tensor]:
        x = x.detach().reshape(-1).long()
        y = y.detach().reshape(-1).long()
        if x.numel() == 0 or x.numel() != y.numel() or num_x <= 1 or num_y <= 1:
            zero = x.new_zeros((), dtype=torch.float32)
            return zero, zero

        x = x.clamp(0, num_x - 1)
        y = y.clamp(0, num_y - 1)
        joint_counts = torch.bincount(x * num_y + y, minlength=num_x * num_y).reshape(num_x, num_y).float()
        total = joint_counts.sum()
        if total <= 0:
            zero = joint_counts.new_zeros(())
            return zero, zero

        p_xy = joint_counts / total
        p_x = p_xy.sum(dim=1)
        p_y = p_xy.sum(dim=0)
        h_x = self._entropy_from_counts(p_x)
        h_y = self._entropy_from_counts(p_y)

        nonzero = p_xy > 0
        expected = (p_x[:, None] * p_y[None, :]).clamp_min(1e-12)
        mi = (p_xy[nonzero] * (p_xy[nonzero].log() - expected[nonzero].log())).sum()
        denom = torch.sqrt((h_x * h_y).clamp_min(1e-12))
        nmi = torch.where((h_x > 0) & (h_y > 0), mi / denom, mi.new_zeros(()))
        return mi, nmi

    def _future_factorized_indices(self, outputs: Dict, batch_size: int) -> Tuple[Tensor | None, Tensor | None]:
        if "radius_indices" not in outputs or "direction_indices" not in outputs:
            return None, None
        radius = outputs["radius_indices"].detach().reshape(-1).long()
        direction = outputs["direction_indices"].detach()
        if radius.numel() < batch_size:
            return None, None
        direction = direction.reshape(radius.numel(), -1).long()
        return radius[-batch_size:], direction[-batch_size:]

    def _action_bins(self, action: Tensor) -> Tuple[Tensor, int, Tensor, int, Tensor, int]:
        action_flat = action.detach().float().reshape(action.shape[0], -1)
        norm = torch.linalg.vector_norm(action_flat, dim=-1)
        norm_bins, num_norm_bins = self._rank_bins(norm, int(os.environ.get("POLAR_MI_NUM_BINS", "16")))

        dominant_dim = action_flat.abs().argmax(dim=-1)
        dominant_value = action_flat.gather(1, dominant_dim[:, None]).squeeze(1)
        direction_bins = dominant_dim * 2 + (dominant_value >= 0).long()
        num_direction_bins = max(1, action_flat.shape[-1] * 2)

        action_bins = norm_bins * num_direction_bins + direction_bins
        num_action_bins = num_norm_bins * num_direction_bins
        return norm_bins, num_norm_bins, direction_bins, num_direction_bins, action_bins, num_action_bins

    def _factorized_quality_metrics(self, outputs: Dict, batch: Dict) -> Tuple:
        if "radius_indices" not in outputs or "direction_indices" not in outputs:
            return ()

        radius = outputs["radius_indices"].detach().reshape(-1).long()
        direction = outputs["direction_indices"].detach().reshape(-1).long()
        pair = self._radius_direction_pair_indices(outputs["radius_indices"], outputs["direction_indices"])
        pair_size = self.factorized_vq_num_radius * self.factorized_vq_num_directions
        logs = (
            ("entropy/radius", self._token_entropy(radius, self.factorized_vq_num_radius)),
            ("entropy/direction", self._token_entropy(direction, self.factorized_vq_num_directions)),
            ("entropy/pair", self._token_entropy(pair, pair_size)),
            ("perplexity/radius", self._token_perplexity(radius, self.factorized_vq_num_radius)),
            ("perplexity/direction", self._token_perplexity(direction, self.factorized_vq_num_directions)),
            ("perplexity/pair", self._token_perplexity(pair, pair_size)),
        )

        batch_size = batch["videos"].shape[0] if "videos" in batch else 0
        radius_future, direction_future = self._future_factorized_indices(outputs, batch_size)
        if radius_future is None or direction_future is None:
            return logs

        if "action" in batch:
            norm_bins, num_norm_bins, direction_bins, num_direction_bins, action_bins, num_action_bins = self._action_bins(
                batch["action"].to(device=radius_future.device)
            )
            mi, nmi = self._discrete_mi_nmi(radius_future, self.factorized_vq_num_radius, norm_bins, num_norm_bins)
            logs = logs + (
                ("mi/radius_action_norm", mi),
                ("nmi/radius_action_norm", nmi),
            )

            direction_flat = direction_future.reshape(-1)
            direction_target = direction_bins[:, None].expand_as(direction_future).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(
                direction_flat,
                self.factorized_vq_num_directions,
                direction_target,
                num_direction_bins,
            )
            logs = logs + (
                ("mi/direction_action_direction", mi),
                ("nmi/direction_action_direction", nmi),
            )

            pair_future = (radius_future[:, None] * self.factorized_vq_num_directions + direction_future).reshape(-1)
            action_target = action_bins[:, None].expand_as(direction_future).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(pair_future, pair_size, action_target, num_action_bins)
            logs = logs + (
                ("mi/pair_action_bin", mi),
                ("nmi/pair_action_bin", nmi),
            )

        if "patches" in outputs and outputs["patches"].shape[0] == batch_size and outputs["patches"].shape[1] >= 2:
            state_change = self._sample_vector_norm(outputs["patches"][:, -1] - outputs["patches"][:, 0])
            state_bins, num_state_bins = self._rank_bins(
                state_change.to(device=radius_future.device),
                int(os.environ.get("POLAR_MI_NUM_BINS", "16")),
            )
            mi, nmi = self._discrete_mi_nmi(
                radius_future,
                self.factorized_vq_num_radius,
                state_bins,
                num_state_bins,
            )
            logs = logs + (
                ("mi/radius_dino_state_change_norm", mi),
                ("nmi/radius_dino_state_change_norm", nmi),
            )

        if "radprog_future_offsets" in batch:
            offset = batch["radprog_future_offsets"].to(device=radius_future.device).reshape(-1).long()
            if offset.numel() == radius_future.numel():
                num_offsets = int(offset.max().detach().cpu().item()) + 1 if offset.numel() > 0 else 1
                mi, nmi = self._discrete_mi_nmi(radius_future, self.factorized_vq_num_radius, offset, num_offsets)
                logs = logs + (
                    ("mi/radius_offset", mi),
                    ("nmi/radius_offset", nmi),
                )
        return logs

    def _future_standard_indices(self, outputs: Dict, batch_size: int, key: str = "indices") -> Tensor | None:
        if key not in outputs or batch_size <= 0:
            return None
        indices = outputs[key].detach()
        if indices.shape[0] < batch_size:
            return None
        return indices.reshape(indices.shape[0], -1).long()[-batch_size:]

    def _standard_token_quality_metrics(
        self,
        outputs: Dict,
        batch: Dict,
        *,
        key: str = "indices",
        prefix: str = "code",
        num_latents: int | None = None,
    ) -> Tuple:
        if key not in outputs:
            return ()
        num_latents = int(num_latents or self.lam_num_latents)
        indices = outputs[key].detach().reshape(-1).long()
        logs = (
            (f"entropy/{prefix}", self._token_entropy(indices, num_latents)),
            (f"perplexity/{prefix}", self._token_perplexity(indices, num_latents)),
        )

        batch_size = batch["videos"].shape[0] if "videos" in batch else 0
        future_indices = self._future_standard_indices(outputs, batch_size, key=key)
        if future_indices is None:
            return logs

        if "action" in batch:
            norm_bins, num_norm_bins, direction_bins, num_direction_bins, action_bins, num_action_bins = self._action_bins(
                batch["action"].to(device=future_indices.device)
            )
            token_flat = future_indices.reshape(-1)

            norm_target = norm_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, norm_target, num_norm_bins)
            logs = logs + (
                (f"mi/{prefix}_action_norm", mi),
                (f"nmi/{prefix}_action_norm", nmi),
            )

            direction_target = direction_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, direction_target, num_direction_bins)
            logs = logs + (
                (f"mi/{prefix}_action_direction", mi),
                (f"nmi/{prefix}_action_direction", nmi),
            )

            action_target = action_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, action_target, num_action_bins)
            logs = logs + (
                (f"mi/{prefix}_action_bin", mi),
                (f"nmi/{prefix}_action_bin", nmi),
            )

        if "patches" in outputs and outputs["patches"].shape[0] == batch_size and outputs["patches"].shape[1] >= 2:
            state_change = self._sample_vector_norm(outputs["patches"][:, -1] - outputs["patches"][:, 0])
            state_bins, num_state_bins = self._rank_bins(
                state_change.to(device=future_indices.device),
                int(os.environ.get("POLAR_MI_NUM_BINS", "16")),
            )
            token_flat = future_indices.reshape(-1)
            state_target = state_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, state_target, num_state_bins)
            logs = logs + (
                (f"mi/{prefix}_dino_state_change_norm", mi),
                (f"nmi/{prefix}_dino_state_change_norm", nmi),
            )

        if "radprog_future_offsets" in batch:
            offset = batch["radprog_future_offsets"].to(device=future_indices.device).reshape(-1).long()
            if offset.numel() == future_indices.shape[0]:
                num_offsets = int(offset.max().detach().cpu().item()) + 1 if offset.numel() > 0 else 1
                token_flat = future_indices.reshape(-1)
                offset_target = offset[:, None].expand_as(future_indices).reshape(-1)
                mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, offset_target, num_offsets)
                logs = logs + (
                    (f"mi/{prefix}_offset", mi),
                    (f"nmi/{prefix}_offset", nmi),
                )
        return logs

    def _vq_codebook_weight(self) -> Tensor:
        if hasattr(self.lam.vq, "codebook_weight"):
            return self.lam.vq.codebook_weight()
        return self.lam.vq.codebook.weight

    def _radprog_enabled(self) -> bool:
        return self.radprog_radial_weight > 0.0 or self.radprog_progress_weight > 0.0

    def _radprog_warmup_scale(self) -> float:
        if self.radprog_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.global_step + 1) / float(self.radprog_warmup_steps))

    def _codebook_radprog_enabled(self) -> bool:
        return self.use_vq and self._radprog_enabled() and self.radprog_codebook_aux_weight > 0.0

    def _validate_hyperbolic_config(self) -> None:
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

    def _masked_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

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

    def _poincare_project(self, point: Tensor) -> Tensor:
        sqrt_c = self.hyperbolic_curvature ** 0.5
        max_norm = (1.0 - self.hyperbolic_eps) / sqrt_c
        return self._clamp_norm(point, max_norm)

    def _poincare_expmap0(self, tangent: Tensor) -> Tensor:
        tangent = self._clamp_norm(tangent, self.hyperbolic_lift_max_norm)
        sqrt_c = self.hyperbolic_curvature ** 0.5
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(self.hyperbolic_eps)
        scale = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
        return self._poincare_project(scale * tangent)

    def _hyperbolic_lift(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        tangent = self._clamp_norm(z, self.hyperbolic_lift_max_norm)
        return self._poincare_expmap0(tangent), tangent

    def _poincare_dist0(self, point: Tensor) -> Tensor:
        sqrt_c = self.hyperbolic_curvature ** 0.5
        norm = point.norm(dim=-1).clamp_max((1.0 - self.hyperbolic_eps) / sqrt_c)
        return 2.0 * torch.atanh(sqrt_c * norm) / sqrt_c

    def _poincare_dist(self, x: Tensor, y: Tensor) -> Tensor:
        c = self.hyperbolic_curvature
        sqrt_c = c ** 0.5
        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        diff2 = ((x - y) * (x - y)).sum(dim=-1)
        denom = ((1.0 - c * x2) * (1.0 - c * y2)).clamp_min(self.hyperbolic_eps)
        z = 1.0 + 2.0 * c * diff2 / denom
        return torch.acosh(z.clamp_min(1.0 + self.hyperbolic_eps)) / sqrt_c

    def _future_latent_metrics(self, z_future: Tensor) -> Tuple:
        latent_r_future = torch.linalg.vector_norm(z_future, dim=-1).mean()
        return (
            ("latent_r_future", latent_r_future),
        )

    def _mean_offdiag_cosine(self, tensor: Tensor) -> Tensor:
        vectors = tensor.detach().float()
        vectors = vectors.reshape(vectors.shape[0], -1)
        if vectors.shape[0] <= 1:
            return vectors.new_zeros(())
        vectors = F.normalize(vectors, dim=-1, eps=1e-6)
        cosine = vectors @ vectors.T
        off_diag = ~torch.eye(vectors.shape[0], device=vectors.device, dtype=torch.bool)
        return cosine[off_diag].mean()

    def _batch_variance(self, tensor: Tensor) -> Tensor:
        vectors = tensor.detach().float().reshape(tensor.shape[0], -1)
        return vectors.var(dim=0, unbiased=False).mean()

    def _latent_action_spread_metrics(self, outputs: Dict) -> Tuple:
        logs = ()
        for name, key in (("emb", "emb"), ("z_q", "z_q")):
            if key not in outputs:
                continue
            tensor = outputs[key]
            logs = logs + (
                (f"cosine/{name}", self._mean_offdiag_cosine(tensor)),
                (f"variance/{name}", self._batch_variance(tensor)),
            )

        if self.use_vq:
            codebook = self._vq_codebook_weight()
            logs = logs + (
                ("cosine/codebook", self._mean_offdiag_cosine(codebook)),
                ("variance/codebook", self._batch_variance(codebook)),
            )
        return logs

    def _mean_vector_norm(self, tensor: Tensor, mask: Tensor | None = None) -> Tensor:
        norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
        if mask is None:
            return norms.mean()
        return self._masked_mean(norms, mask)

    def _sample_vector_norm(self, tensor: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
        if norms.ndim == 1:
            return norms
        return norms.reshape(norms.shape[0], -1).mean(dim=1)

    def _pearson_corr(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        x = x[mask].float()
        y = y[mask].float()
        if x.numel() <= 1:
            return x.new_zeros(())
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        if denom <= 1e-12:
            return x.new_zeros(())
        return (x * y).sum() / denom

    def _offset_norm_logs(
        self,
        name: str,
        z_self: Tensor,
        z_mid: Tensor,
        z_future: Tensor,
        batch: Dict,
    ) -> Tuple:
        required = ("radprog_mid_offsets", "radprog_future_offsets", "radprog_valid")
        if any(key not in batch for key in required):
            return ()

        batch_size = batch["radprog_mid_offsets"].shape[0]
        if z_self.shape[0] != batch_size or z_mid.shape[0] != batch_size or z_future.shape[0] != batch_size:
            return ()

        device = z_self.device
        mid_offsets = batch["radprog_mid_offsets"].to(device=device, dtype=torch.float32)
        future_offsets = batch["radprog_future_offsets"].to(device=device, dtype=torch.float32)
        valid = batch["radprog_valid"].to(device=device, dtype=torch.bool)

        offsets = torch.stack(
            [
                torch.zeros_like(mid_offsets),
                mid_offsets,
                future_offsets,
            ],
            dim=1,
        )
        norms = torch.stack(
            [
                self._sample_vector_norm(z_self),
                self._sample_vector_norm(z_mid),
                self._sample_vector_norm(z_future),
            ],
            dim=1,
        )
        point_mask = valid[:, None].expand_as(norms)

        logs = ((f"offset_norm/{name}/pearson", self._pearson_corr(offsets, norms, point_mask)),)
        if not point_mask.any():
            return logs

        max_offset = int(offsets[point_mask].max().detach().cpu().item())
        offset_ids = offsets.round().long()
        for offset in range(max_offset + 1):
            offset_mask = point_mask & (offset_ids == offset)
            if offset_mask.any():
                logs = logs + ((f"offset_norm/{name}/k_{offset}", norms[offset_mask].mean()),)
        return logs

    def _offset_norm_metrics(self, outputs: Dict, batch: Dict) -> Tuple:
        logs = ()
        groups = (
            ("emb", ("radprog_z_self", "radprog_z_mid", "radprog_z_future")),
            ("z_q", ("radprog_zq_self", "radprog_zq_mid", "radprog_zq_future")),
            ("z", ("radprog_code_self", "radprog_code_mid", "radprog_code_future")),
        )
        for name, keys in groups:
            if all(key in outputs for key in keys):
                logs = logs + self._offset_norm_logs(
                    name,
                    outputs[keys[0]],
                    outputs[keys[1]],
                    outputs[keys[2]],
                    batch,
                )
        return logs

    def _latent_action_norm_metrics(self, outputs: Dict, batch: Dict) -> Tuple:
        logs = ()
        for key in ("z", "emb", "emb_raw", "z_q"):
            if key in outputs:
                logs = logs + ((f"latent_action/{key}_norm", self._mean_vector_norm(outputs[key])),)

        radprog_keys = {
            "self": "radprog_z_self",
            "mid": "radprog_z_mid",
            "future": "radprog_z_future",
        }
        if not all(key in outputs for key in radprog_keys.values()):
            return logs

        valid = batch.get("radprog_valid")
        if valid is not None:
            sample = outputs["radprog_z_future"]
            valid = valid.to(device=sample.device, dtype=torch.float32)[:, None]
            valid = valid.expand_as(sample[..., 0])

        for name, key in radprog_keys.items():
            logs = logs + (
                (f"radprog/latent_norm_{name}", self._mean_vector_norm(outputs[key], valid)),
            )

        if self.hyperbolic_latent_enabled:
            for name, key in radprog_keys.items():
                prelift = outputs[key].detach().float()
                logs = logs + (
                    (f"radprog/prelift_norm_{name}", self._mean_vector_norm(prelift, valid)),
                )

        logs = logs + self._offset_norm_metrics(outputs, batch)
        return logs

    def _log_metric_name(self, name: str) -> str:
        name_map = {
            "mse_loss": "train/mse_loss",
            "q_loss": "train/q_loss",
            "commit_loss": "train/commit_loss",
            "mse_mid": "loss/mse_mid",
            "mse_future": "loss/mse_future",
            "radprog_radial_loss": "loss/radprog_radial",
            "radprog_progress_loss": "loss/radprog_progress",
            "radprog_codebook_radial_loss": "loss/radprog_codebook_radial",
            "radprog_codebook_progress_loss": "loss/radprog_codebook_progress",
            "radprog_warmup_scale": "radprog/warmup_scale",
            "code_usage": "train/code_usage",
            "radius_code_usage": "train/radius_code_usage",
            "direction_code_usage": "train/direction_code_usage",
            "radius_direction_pair_usage": "train/radius_direction_pair_usage",
            "latent_r_future": "scale/latent_norm_future",
            "radprog_hyperbolic_enabled": "config/hyperbolic_enabled",
            "radprog_codebook_hyperbolic_enabled": "config/hyperbolic_codebook_enabled",
            "radprog_valid_frac": "radprog/valid_frac",
            "radprog_codebook_valid_frac": "radprog/codebook/valid_frac",
            "radprog_radial_order_frac": "radprog/order/radial_frac",
            "radprog_norm_order_frac": "radprog/order/norm_frac",
            "radprog_codebook_radial_order_frac": "radprog/codebook/order/radial_frac",
            "radprog_codebook_norm_order_frac": "radprog/codebook/order/norm_frac",
            "radprog_d_mid": "scale/radprog/d_mid",
            "radprog_d_future": "scale/radprog/d_future",
            "radprog_r_self": "scale/radprog/r_self",
            "radprog_r_mid": "scale/radprog/r_mid",
            "radprog_r_future": "scale/radprog/r_future",
            "radprog_codebook_d_mid": "scale/radprog/codebook/d_mid",
            "radprog_codebook_d_future": "scale/radprog/codebook/d_future",
            "radprog_codebook_r_self": "scale/radprog/codebook/r_self",
            "radprog_codebook_r_mid": "scale/radprog/codebook/r_mid",
            "radprog_codebook_r_future": "scale/radprog/codebook/r_future",
            "radprog_prelift_norm_self": "scale/radprog/prelift_norm_self",
            "radprog_prelift_norm_mid": "scale/radprog/prelift_norm_mid",
            "radprog_prelift_norm_future": "scale/radprog/prelift_norm_future",
            "radprog_codebook_prelift_norm_self": "scale/radprog/codebook/prelift_norm_self",
            "radprog_codebook_prelift_norm_mid": "scale/radprog/codebook/prelift_norm_mid",
            "radprog_codebook_prelift_norm_future": "scale/radprog/codebook/prelift_norm_future",
            "radprog_zq_radial_order_frac": "radprog/z_q/order/radial_frac",
            "radprog_zq_norm_order_frac": "radprog/z_q/order/norm_frac",
            "radprog_zq_d_mid": "scale/radprog/z_q/d_mid",
            "radprog_zq_d_future": "scale/radprog/z_q/d_future",
            "radprog_zq_r_self": "scale/radprog/z_q/r_self",
            "radprog_zq_r_mid": "scale/radprog/z_q/r_mid",
            "radprog_zq_r_future": "scale/radprog/z_q/r_future",
            "radprog_zq_prelift_norm_self": "scale/radprog/z_q/prelift_norm_self",
            "radprog_zq_prelift_norm_mid": "scale/radprog/z_q/prelift_norm_mid",
            "radprog_zq_prelift_norm_future": "scale/radprog/z_q/prelift_norm_future",
        }
        if name in name_map:
            return name_map[name]
        if name.startswith("probe/"):
            return name
        if name.startswith("entropy/"):
            return name
        if name.startswith("perplexity/"):
            return name
        if name.startswith("mi/"):
            return name
        if name.startswith("nmi/"):
            return name
        if name.startswith("cosine/"):
            return f"scale/{name}"
        if name.startswith("variance/"):
            return f"scale/{name}"
        if name.startswith("latent_action/"):
            return f"scale/{name}"
        if name.startswith("offset_norm/"):
            return f"scale/{name}"
        if name.startswith("radprog/latent_norm_"):
            return f"scale/{name}"
        if name.startswith("radprog/prelift_norm_"):
            return f"scale/{name}"
        if name.startswith("radprog/"):
            return name
        return f"misc/{name}"

    def _build_action_probe(self, z_dim: int, action_dim: int, device: torch.device) -> None:
        probe = torch.nn.Linear(z_dim, action_dim).to(device)
        self._action_probe_state["probe"] = probe
        self._action_probe_state["optimizer"] = torch.optim.Adam(probe.parameters(), lr=self.action_probe_lr)
        if self.action_probe_shuffle:
            shuffled = torch.nn.Linear(z_dim, action_dim).to(device)
            self._action_probe_state["shuffled"] = shuffled
            self._action_probe_state["shuffled_optimizer"] = torch.optim.Adam(
                shuffled.parameters(),
                lr=self.action_probe_lr,
            )

    def _run_action_probe(
        self,
        probe: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        z_flat: Tensor,
        target: Tensor,
        name: str,
        shuffle: bool = False,
    ) -> Tuple:
        if shuffle:
            target = target[torch.randperm(z_flat.shape[0], device=z_flat.device)]

        pred = probe(z_flat)
        probe_loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        probe_loss.backward()
        optimizer.step()

        return ((f"probe/{name}/l2", probe_loss.detach()),)

    def _action_probe_step(self, outputs: Dict, batch: Dict) -> Tuple:
        if "action" not in batch:
            return ()

        batch_size = batch["action"].shape[0]
        if "z_q" in outputs:
            z_future = outputs["z_q"]
        else:
            z_future = outputs.get("radprog_z_future", outputs["emb"])
        if z_future.shape[0] == batch_size and z_future.ndim >= 4:
            z_future = z_future[:, -1]
        if z_future.shape[0] != batch_size:
            z_future = z_future.reshape(batch_size, -1, *z_future.shape[1:])[:, -1]

        z_flat = z_future.detach().reshape(batch_size, -1).float()
        target = batch["action"].to(device=z_flat.device, dtype=z_flat.dtype).reshape(batch_size, -1)
        probe_state = self._action_probe_state
        if probe_state["probe"] is None:
            self._build_action_probe(z_flat.shape[-1], target.shape[-1], z_flat.device)
            probe_state = self._action_probe_state

        logs = self._run_action_probe(
            probe_state["probe"],
            probe_state["optimizer"],
            z_flat,
            target,
            "z_future_to_action",
        )
        if probe_state["shuffled"] is not None:
            logs = logs + self._run_action_probe(
                probe_state["shuffled"],
                probe_state["shuffled_optimizer"],
                z_flat,
                target,
                "z_future_to_action_shuffled",
                shuffle=True,
            )
        return logs

    @staticmethod
    def _drop_action_probe_checkpoint_state(checkpoint: Dict) -> None:
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key in list(state_dict.keys()):
            if key.startswith("action_probe") or key.startswith("_action_probe"):
                state_dict.pop(key)

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        self._drop_action_probe_checkpoint_state(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        self._drop_action_probe_checkpoint_state(checkpoint)

    def _compute_radprog_losses(
        self,
        outputs: Dict,
        batch: Dict,
        output_keys: Tuple[str, str, str] = ("radprog_z_self", "radprog_z_mid", "radprog_z_future"),
        metric_prefix: str = "radprog",
    ) -> Tuple[Tensor, Tensor, Tuple]:
        batch_keys = ("radprog_mid_offsets", "radprog_future_offsets", "radprog_valid")
        missing = [key for key in output_keys if key not in outputs]
        missing.extend(key for key in batch_keys if key not in batch)
        if missing:
            raise ValueError(
                f"{metric_prefix} is enabled, but required keys are missing: "
                + ", ".join(sorted(missing))
            )

        z_self = outputs[output_keys[0]]
        z_mid = outputs[output_keys[1]]
        z_future = outputs[output_keys[2]]
        if z_self.shape != z_mid.shape or z_self.shape != z_future.shape:
            raise ValueError(
                f"{metric_prefix} latent shapes must match, got "
                f"self={tuple(z_self.shape)}, mid={tuple(z_mid.shape)}, future={tuple(z_future.shape)}"
            )
        metric_name = "radprog" if metric_prefix == "radprog" else metric_prefix

        compute_dtype = torch.float32 if self.hyperbolic_latent_enabled else z_self.dtype
        z_self = z_self.to(dtype=compute_dtype)
        z_mid = z_mid.to(dtype=compute_dtype)
        z_future = z_future.to(dtype=compute_dtype)
        mid_offsets = batch["radprog_mid_offsets"].to(device=z_self.device, dtype=compute_dtype)[:, None]
        future_offsets = batch["radprog_future_offsets"].to(device=z_self.device, dtype=compute_dtype)[:, None]
        valid = batch["radprog_valid"].to(device=z_self.device, dtype=compute_dtype)[:, None]
        valid = valid.expand_as(z_self[..., 0])

        logs = ()
        if self.hyperbolic_latent_enabled:
            z_self_h, z_self_t = self._hyperbolic_lift(z_self)
            z_mid_h, z_mid_t = self._hyperbolic_lift(z_mid)
            z_future_h, z_future_t = self._hyperbolic_lift(z_future)
            d_mid = torch.nan_to_num(self._poincare_dist(z_self_h, z_mid_h), nan=0.0, posinf=1e4, neginf=0.0)
            d_future = torch.nan_to_num(self._poincare_dist(z_self_h, z_future_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_self = torch.nan_to_num(self._poincare_dist0(z_self_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_mid = torch.nan_to_num(self._poincare_dist0(z_mid_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_future = torch.nan_to_num(self._poincare_dist0(z_future_h), nan=0.0, posinf=1e4, neginf=0.0)
            logs = logs + (
                (f"{metric_name}_hyperbolic_enabled", z_self.new_ones(())),
                (f"{metric_name}_prelift_norm_self", self._masked_mean(z_self_t.norm(dim=-1), valid)),
                (f"{metric_name}_prelift_norm_mid", self._masked_mean(z_mid_t.norm(dim=-1), valid)),
                (f"{metric_name}_prelift_norm_future", self._masked_mean(z_future_t.norm(dim=-1), valid)),
            )
        else:
            d_mid = torch.linalg.vector_norm(z_mid - z_self, dim=-1)
            d_future = torch.linalg.vector_norm(z_future - z_self, dim=-1)
            r_self = torch.linalg.vector_norm(z_self, dim=-1)
            r_mid = torch.linalg.vector_norm(z_mid, dim=-1)
            r_future = torch.linalg.vector_norm(z_future, dim=-1)
            logs = logs + ((f"{metric_name}_hyperbolic_enabled", z_self.new_zeros(())),)

        radial_loss = self._masked_mean(F.softplus((d_mid - d_future).clamp(min=-50.0, max=50.0)), valid)

        first_leg_arg = self.radprog_progress_alpha * mid_offsets + r_self - r_mid
        second_leg_arg = self.radprog_progress_alpha * (future_offsets - mid_offsets) + r_mid - r_future
        first_leg = F.softplus(first_leg_arg.clamp(min=-50.0, max=50.0))
        second_leg = F.softplus(second_leg_arg.clamp(min=-50.0, max=50.0))
        progress_loss = self._masked_mean(first_leg + second_leg, valid)

        logs = logs + (
            (f"{metric_name}_radial_loss", radial_loss),
            (f"{metric_name}_progress_loss", progress_loss),
            (f"{metric_name}_valid_frac", valid.mean()),
            (f"{metric_name}_radial_order_frac", self._masked_mean((d_future > d_mid).to(d_mid.dtype), valid)),
            (
                f"{metric_name}_norm_order_frac",
                self._masked_mean(((r_mid > r_self) & (r_future > r_mid)).to(r_self.dtype), valid),
            ),
            (f"{metric_name}_d_mid", self._masked_mean(d_mid, valid)),
            (f"{metric_name}_d_future", self._masked_mean(d_future, valid)),
            (f"{metric_name}_r_self", self._masked_mean(r_self, valid)),
            (f"{metric_name}_r_mid", self._masked_mean(r_mid, valid)),
            (f"{metric_name}_r_future", self._masked_mean(r_future, valid)),
        )
        return radial_loss, progress_loss, logs

    def _compute_radprog_zq_diagnostics(self, outputs: Dict, batch: Dict) -> Tuple:
        output_keys = ("radprog_zq_self", "radprog_zq_mid", "radprog_zq_future")
        batch_keys = ("radprog_mid_offsets", "radprog_future_offsets", "radprog_valid")
        if any(key not in outputs for key in output_keys) or any(key not in batch for key in batch_keys):
            return ()

        z_self = outputs["radprog_zq_self"].detach()
        z_mid = outputs["radprog_zq_mid"].detach()
        z_future = outputs["radprog_zq_future"].detach()
        if z_self.shape != z_mid.shape or z_self.shape != z_future.shape:
            raise ValueError(
                "RadProg z_q diagnostic shapes must match, got "
                f"self={tuple(z_self.shape)}, mid={tuple(z_mid.shape)}, future={tuple(z_future.shape)}"
            )

        compute_dtype = torch.float32 if self.hyperbolic_latent_enabled else z_self.dtype
        z_self = z_self.to(dtype=compute_dtype)
        z_mid = z_mid.to(dtype=compute_dtype)
        z_future = z_future.to(dtype=compute_dtype)
        valid = batch["radprog_valid"].to(device=z_self.device, dtype=compute_dtype)[:, None]
        valid = valid.expand_as(z_self[..., 0])

        logs = ()
        if self.hyperbolic_latent_enabled:
            z_self_h, z_self_t = self._hyperbolic_lift(z_self)
            z_mid_h, z_mid_t = self._hyperbolic_lift(z_mid)
            z_future_h, z_future_t = self._hyperbolic_lift(z_future)
            d_mid = torch.nan_to_num(self._poincare_dist(z_self_h, z_mid_h), nan=0.0, posinf=1e4, neginf=0.0)
            d_future = torch.nan_to_num(self._poincare_dist(z_self_h, z_future_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_self = torch.nan_to_num(self._poincare_dist0(z_self_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_mid = torch.nan_to_num(self._poincare_dist0(z_mid_h), nan=0.0, posinf=1e4, neginf=0.0)
            r_future = torch.nan_to_num(self._poincare_dist0(z_future_h), nan=0.0, posinf=1e4, neginf=0.0)
            logs = logs + (
                ("radprog_zq_prelift_norm_self", self._masked_mean(z_self_t.norm(dim=-1), valid)),
                ("radprog_zq_prelift_norm_mid", self._masked_mean(z_mid_t.norm(dim=-1), valid)),
                ("radprog_zq_prelift_norm_future", self._masked_mean(z_future_t.norm(dim=-1), valid)),
            )
        else:
            d_mid = torch.linalg.vector_norm(z_mid - z_self, dim=-1)
            d_future = torch.linalg.vector_norm(z_future - z_self, dim=-1)
            r_self = torch.linalg.vector_norm(z_self, dim=-1)
            r_mid = torch.linalg.vector_norm(z_mid, dim=-1)
            r_future = torch.linalg.vector_norm(z_future, dim=-1)

        logs = logs + (
            ("radprog_zq_radial_order_frac", self._masked_mean((d_future > d_mid).to(d_mid.dtype), valid)),
            (
                "radprog_zq_norm_order_frac",
                self._masked_mean(((r_mid > r_self) & (r_future > r_mid)).to(r_self.dtype), valid),
            ),
            ("radprog_zq_d_mid", self._masked_mean(d_mid, valid)),
            ("radprog_zq_d_future", self._masked_mean(d_future, valid)),
            ("radprog_zq_r_self", self._masked_mean(r_self, valid)),
            ("radprog_zq_r_mid", self._masked_mean(r_mid, valid)),
            ("radprog_zq_r_future", self._masked_mean(r_future, valid)),
        )
        return logs

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        loss = mse_loss
        loss_logs = (("mse_loss", mse_loss),)
        if self.use_vq:
            q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
            commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()
            loss = loss + q_loss + self.vq_beta * commit_loss
            code_usage = self._code_usage(outputs["indices"], self.lam_num_latents)
            loss_logs = loss_logs + (
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("code_usage", code_usage),
            )
            if "radius_indices" in outputs and "direction_indices" in outputs:
                loss_logs = loss_logs + (
                    (
                        "radius_code_usage",
                        self._code_usage(outputs["radius_indices"], self.factorized_vq_num_radius),
                    ),
                    (
                        "direction_code_usage",
                        self._code_usage(outputs["direction_indices"], self.factorized_vq_num_directions),
                    ),
                    (
                        "radius_direction_pair_usage",
                        self._radius_direction_pair_usage(outputs["radius_indices"], outputs["direction_indices"]),
                    ),
                )
                loss_logs = loss_logs + self._factorized_quality_metrics(outputs, batch)
            else:
                loss_logs = loss_logs + self._standard_token_quality_metrics(outputs, batch)
        if (
            "radprog_mid_pixel_values" in batch
            and gt_future_frames.ndim >= 3
            and gt_future_frames.shape[1] == 2
            and outputs["recon"].shape[1] == 2
        ):
            loss_logs = loss_logs + (
                ("mse_mid", ((gt_future_frames[:, 0] - outputs["recon"][:, 0]) ** 2).mean()),
                ("mse_future", ((gt_future_frames[:, 1] - outputs["recon"][:, 1]) ** 2).mean()),
            )
        loss_logs = loss_logs + self._future_latent_metrics(outputs["emb"])
        loss_logs = loss_logs + self._latent_action_spread_metrics(outputs)
        loss_logs = loss_logs + self._latent_action_norm_metrics(outputs, batch)
        loss_logs = loss_logs + self._compute_radprog_zq_diagnostics(outputs, batch)
        if self._radprog_enabled():
            radial_loss, progress_loss, radprog_logs = self._compute_radprog_losses(outputs, batch)
            radprog_warmup_scale = self._radprog_warmup_scale()
            loss = (
                loss
                + radprog_warmup_scale
                * (self.radprog_radial_weight * radial_loss + self.radprog_progress_weight * progress_loss)
            )
            loss_logs = loss_logs + radprog_logs + (
                ("radprog_warmup_scale", radial_loss.new_tensor(radprog_warmup_scale)),
            )
            if self._codebook_radprog_enabled():
                code_radial_loss, code_progress_loss, code_radprog_logs = self._compute_radprog_losses(
                    outputs,
                    batch,
                    output_keys=("radprog_code_self", "radprog_code_mid", "radprog_code_future"),
                    metric_prefix="radprog_codebook",
                )
                loss = (
                    loss
                    + radprog_warmup_scale
                    * self.radprog_codebook_aux_weight
                    * (
                        self.radprog_radial_weight * code_radial_loss
                        + self.radprog_progress_weight * code_progress_loss
                    )
                )
                loss_logs = loss_logs + code_radprog_logs

        return outputs, loss, loss_logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            shared_start = time.perf_counter()
        outputs, loss, aux_losses = self.shared_step(batch)
        if self.action_probe_enabled:
            aux_losses = aux_losses + self._action_probe_step(outputs, batch)
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._profile_shared_step = time.perf_counter() - shared_start

        logs = {self._log_metric_name(k): v for k, v in aux_losses}
        # Keep W&B scalar series single-valued; on_epoch=True creates duplicate *_step/*_epoch metrics.
        self.log(
            "loss/total",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log_dict(
            logs,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        return loss

    def on_train_start(self) -> None:
        if self._profile_steps:
            self._profile_prev_batch_end = time.perf_counter()

    def _profile_to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def _extract_input_profile(self, batch: Dict) -> Dict[str, float]:
        profile = batch.get("_profile") if isinstance(batch, dict) else None
        if not isinstance(profile, dict):
            return {}
        return {key: self._profile_to_float(value) for key, value in profile.items()}

    def _reduce_profile_values(self, values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        if not values:
            return {}
        keys = list(values.keys())
        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        local = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)

        max_values = local.clone()
        mean_values = local.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(max_values, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(mean_values, op=torch.distributed.ReduceOp.SUM)
            mean_values /= torch.distributed.get_world_size()

        return {
            key: {
                "local": float(local[idx].detach().cpu().item()),
                "max": float(max_values[idx].detach().cpu().item()),
                "mean": float(mean_values[idx].detach().cpu().item()),
            }
            for idx, key in enumerate(keys)
        }

    def _format_profile_triplet(self, reduced: Dict[str, Dict[str, float]], key: str, label: str) -> str:
        values = reduced.get(key, {"local": 0.0, "max": 0.0, "mean": 0.0})
        return (
            f"{label}={values['local']:.3f}s "
            f"{label}_max={values['max']:.3f}s "
            f"{label}_mean={values['mean']:.3f}s"
        )

    def on_before_batch_transfer(self, batch: Dict, dataloader_idx: int) -> Dict:
        if self._profile_steps:
            self._profile_batch_transfer_start = time.perf_counter()
        return batch

    def on_after_batch_transfer(self, batch: Dict, dataloader_idx: int) -> Dict:
        if self._profile_steps and self._profile_batch_transfer_start is not None:
            self._profile_batch_transfer = time.perf_counter() - self._profile_batch_transfer_start
        return batch

    def on_train_batch_start(self, batch: Dict, batch_idx: int) -> None:
        if not self._profile_steps or batch_idx >= self._profile_steps:
            return
        now = time.perf_counter()
        self._profile_batch_start = now
        self._profile_active_batch = True
        self._profile_input = self._extract_input_profile(batch)
        if self._profile_prev_batch_end is not None:
            self._profile_data_wait = now - self._profile_prev_batch_end

    def on_before_backward(self, loss: Tensor) -> None:
        if not self._profile_steps or not self._profile_active_batch:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profile_backward_start = time.perf_counter()

    def on_after_backward(self) -> None:
        if not self._profile_steps or not self._profile_active_batch or self._profile_backward_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profile_backward = time.perf_counter() - self._profile_backward_start

    def _maybe_restart_dead_codes(self) -> None:
        if not self.use_vq:
            return
        if self.vq_restart_interval_steps <= 0:
            return
        if self.global_step <= 0 or self.global_step % self.vq_restart_interval_steps != 0:
            return
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_train_batch_end(self, outputs: Tensor, batch: Dict, batch_idx: int) -> None:
        self._maybe_restart_dead_codes()
        if not self._profile_steps:
            return
        if batch_idx < self._profile_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            profile_values = {
                "data_wait": self._profile_data_wait,
                "batch_transfer": self._profile_batch_transfer,
                "host_fetch": max(0.0, self._profile_data_wait - self._profile_batch_transfer),
                **self._profile_input,
            }
            if self._profile_input:
                measured_input = (
                    self._profile_input.get("rlds_next_sum", 0.0)
                    + self._profile_input.get("batch_transform_sum", 0.0)
                    + self._profile_input.get("collate_total", 0.0)
                )
                profile_values["input_unattributed"] = (
                    profile_values["host_fetch"] - measured_input
                )
            reduced_profile = self._reduce_profile_values(profile_values)
            if not self.distributed_state.is_main_process:
                self._profile_prev_batch_end = now
                self._profile_active_batch = False
                return
            total = now - self._profile_batch_start if self._profile_batch_start is not None else 0.0
            print(
                "LAM_PROFILE "
                f"step={batch_idx + 1:06d} "
                f"data_wait={self._profile_data_wait:.3f}s "
                f"shared_step={self._profile_shared_step:.3f}s "
                f"backward={self._profile_backward:.3f}s "
                f"total={total:.3f}s",
                flush=True,
            )
            print(
                "LAM_DATA_PROFILE "
                f"step={batch_idx + 1:06d} "
                f"{self._format_profile_triplet(reduced_profile, 'data_wait', 'data_wait')} "
                f"{self._format_profile_triplet(reduced_profile, 'host_fetch', 'host_fetch')} "
                f"{self._format_profile_triplet(reduced_profile, 'batch_transfer', 'batch_transfer')} "
                f"{self._format_profile_triplet(reduced_profile, 'rlds_next_sum', 'rlds_next_sum')} "
                f"{self._format_profile_triplet(reduced_profile, 'rlds_next_item_max', 'rlds_next_item_max')} "
                f"{self._format_profile_triplet(reduced_profile, 'batch_transform_sum', 'batch_transform_sum')} "
                f"{self._format_profile_triplet(reduced_profile, 'batch_transform_item_max', 'batch_transform_item_max')} "
                f"{self._format_profile_triplet(reduced_profile, 'collate_total', 'collate_total')} "
                f"{self._format_profile_triplet(reduced_profile, 'collate_radprog', 'collate_radprog')} "
                f"{self._format_profile_triplet(reduced_profile, 'input_unattributed', 'input_unattributed')}",
                flush=True,
            )
            self._profile_prev_batch_end = now
        self._profile_active_batch = False

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, aux_losses = self.shared_step(batch)
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if not self.use_vq:
            return
        if self.vq_restart_interval_steps > 0:
            return
        max_steps = getattr(self.trainer, "max_steps", None)
        if max_steps is not None and max_steps > 0 and self.global_step >= max_steps:
            return
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if not self.use_vq:
            return
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self._vq_codebook_weight()[top_indices]
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
