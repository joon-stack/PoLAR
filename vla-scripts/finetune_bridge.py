import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
import yaml
from ema_pytorch import EMA
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import (
    IGNORE_INDEX,
    PaddedCollatorForWindowActionPrediction,
    select_lam_token_indices,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import (
    LeRobotWindowCacheDataset,
    RLDSBatchTransformWindowActionWithHistory,
    RLDSDataset,
)
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_POLAR_LAM_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "latent_action_model"
    / "config"
    / "polar_tokenizer_bridge.yaml"
)
DEFAULT_POLAR_LAM_CHECKPOINT = Path(__file__).resolve().parents[1] / "weights" / "polar_tokenizer_bridge.ckpt"


from prismatic.models.policy.transformer_utils import MAPBlock

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512, latent_action_token_len = 5):
        super().__init__()
        self.latent_action_token_len = latent_action_token_len
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        if latent_action_tokens.shape[1] < self.latent_action_token_len:
            raise ValueError(
                "Not enough latent action tokens: "
                f"expected at least {self.latent_action_token_len}, got {latent_action_tokens.shape[1]}."
            )
        latent_action_tokens = latent_action_tokens[:, -self.latent_action_token_len:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla = False, window_size = 12, latent_action_token_len = 5):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size, latent_action_token_len=latent_action_token_len)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        visual_embed = vla_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches ].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens



@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/path/to/your/pretrained-polar-vla"            # Path to your local PoLAR latent VLA path
    lam_path: str = str(DEFAULT_POLAR_LAM_CHECKPOINT)
    # Directory Paths
    data_root_dir: Path = Path("/path/to/your/rlds_data_collection")
    dataset_name: str = "bridge"
    dataset_format: str = "rlds"
    lerobot_cache_dir: Optional[Path] = None
    run_root_dir: Path = Path("runs/finetune_bridge")
    adapter_tmp_dir: Path = Path("runs/finetune_bridge_adapter_tmp")

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 30000                                          # Max number of fine-tuning steps
    save_steps: int = 30000                                         # Interval for checkpoint saving
    learning_rate: float = 3.5e-4                                   # Fine-tuning learning rate
    use_scheduler: bool = True                                      # Whether to use StepLR
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 16000                                # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # LAM setting
    lam_kind: str = "visual_vq_factorized"
    lam_config_path: Optional[Path] = DEFAULT_POLAR_LAM_CONFIG
    lam_token_view: str = "indices"
    latent_action_token_len: int = 5
    window_size: int = 12
        
    # LoRA Arguments
    freeze_vla: bool = False
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "polar_finetune_bridge"
    wandb_entity: Optional[str] = None
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases



def _load_lam_state_dict(lam_path: str) -> dict:
    lam_ckpt = torch.load(lam_path, map_location="cpu")["state_dict"]
    return {key.replace("lam.", ""): value for key, value in lam_ckpt.items()}


def _expected_lam_token_len(lam_vq_type: str, lam_num_codes: int, lam_token_view: str) -> int:
    lam_token_view = (lam_token_view or "indices").strip().lower()
    if lam_token_view == "indices":
        return lam_num_codes + 1 if lam_vq_type == "factorized" else lam_num_codes
    if lam_token_view == "factorized_direction":
        if lam_vq_type != "factorized":
            raise ValueError("lam_token_view='factorized_direction' requires vq_type='factorized'.")
        return lam_num_codes
    raise ValueError(f"Unsupported lam_token_view={lam_token_view!r}.")


def _build_latent_action_model(cfg: FinetuneConfig) -> torch.nn.Module:
    if cfg.lam_kind != "visual_vq_factorized":
        raise ValueError("Only lam_kind='visual_vq_factorized' is supported in the public PoLAR path.")
    if cfg.lam_config_path is None:
        raise ValueError("lam_config_path is required for PoLAR visual VQ LAM loading.")

    from latent_action_model.genie.modules.lam_visual_vq import VisualVQDINOLatentActionModel

    with open(cfg.lam_config_path, "r") as f:
        lam_model_cfg = yaml.safe_load(f)["model"]
    lam_vq_type = str(lam_model_cfg.get("vq_type", "standard")).strip().lower()
    lam_num_codes = int(lam_model_cfg.get("lam_num_codes", 4))
    expected_token_len = _expected_lam_token_len(lam_vq_type, lam_num_codes, cfg.lam_token_view)
    if cfg.latent_action_token_len > 0 and cfg.latent_action_token_len != expected_token_len:
        raise ValueError(
            "latent_action_token_len must match the LAM config: "
            f"got {cfg.latent_action_token_len}, expected {expected_token_len} "
            f"for vq_type={lam_vq_type!r}, lam_num_codes={lam_num_codes}, "
            f"lam_token_view={cfg.lam_token_view!r}."
        )

    latent_action_model = VisualVQDINOLatentActionModel(
        in_dim=lam_model_cfg.get("image_channels", 3),
        model_dim=lam_model_cfg["lam_model_dim"],
        latent_dim=lam_model_cfg["lam_latent_dim"],
        num_latents=lam_model_cfg["lam_num_latents"],
        num_codes=lam_num_codes,
        patch_size=lam_model_cfg["lam_patch_size"],
        enc_blocks=lam_model_cfg["lam_enc_blocks"],
        dec_blocks=lam_model_cfg["lam_dec_blocks"],
        num_heads=lam_model_cfg["lam_num_heads"],
        dropout=lam_model_cfg.get("lam_dropout", 0.),
        hyperbolic_action_prelift_enabled=lam_model_cfg.get("hyperbolic_latent_enabled", False),
        hyperbolic_curvature=lam_model_cfg.get("hyperbolic_curvature", 1.0),
        hyperbolic_prelift_mode=lam_model_cfg.get("hyperbolic_prelift_mode", "none"),
        hyperbolic_prelift_scale=lam_model_cfg.get("hyperbolic_prelift_scale", 1.0),
        hyperbolic_tangent_max_norm=lam_model_cfg.get("hyperbolic_tangent_max_norm", 0.0),
        hyperbolic_lift_max_norm=lam_model_cfg.get("hyperbolic_lift_max_norm", 0.0),
        hyperbolic_eps=lam_model_cfg.get("hyperbolic_eps", 1e-5),
        latent_output_global_scale=lam_model_cfg.get("latent_output_global_scale", 1.0),
        use_vq=lam_model_cfg.get("use_vq", True),
        vq_type=lam_vq_type,
        factorized_vq_num_radius=lam_model_cfg.get("factorized_vq_num_radius", 4),
        factorized_vq_num_directions=lam_model_cfg.get("factorized_vq_num_directions", 4),
        factorized_vq_radius_values=lam_model_cfg.get("factorized_vq_radius_values"),
    )
    latent_action_model.load_state_dict(_load_lam_state_dict(cfg.lam_path), strict=True)
    return latent_action_model

def _materialize_lam_tokens_in_train_loop(
    batch: dict,
    latent_action_model: torch.nn.Module,
    tokenizer,
    prompt_builder_fn,
    model_max_length: int,
    pad_token_id: int,
    latent_action_token_len: int,
    lam_token_view: str,
    device_id: int,
    predict_stop_token: bool = True,
) -> dict:
    initial = batch.pop("lam_initial_pixel_values").to(device_id, non_blocking=True)
    target = batch.pop("lam_target_pixel_values").to(device_id, non_blocking=True)
    video = torch.stack([initial, target], dim=1)
    with torch.inference_mode():
        outputs = latent_action_model.vq_encode(video)
        latent_action_idx = select_lam_token_indices(outputs, lam_token_view, initial.shape[0]).cpu()

    has_history = batch.pop("has_history")
    hist_action_idx = torch.zeros_like(latent_action_idx)
    hist_indices = torch.nonzero(has_history, as_tuple=False).flatten().tolist()
    if hist_indices:
        hist_initial_all = batch.pop("hist_lam_initial_pixel_values")
        hist_target_all = batch.pop("hist_lam_target_pixel_values")
        hist_initial = hist_initial_all[hist_indices].to(device_id, non_blocking=True)
        hist_target = hist_target_all[hist_indices].to(device_id, non_blocking=True)
        hist_video = torch.stack([hist_initial, hist_target], dim=1)
        with torch.inference_mode():
            outputs = latent_action_model.vq_encode(hist_video)
            encoded_hist = select_lam_token_indices(outputs, lam_token_view, len(hist_indices)).cpu()
        hist_action_idx[hist_indices] = encoded_hist
    else:
        batch.pop("hist_lam_initial_pixel_values")
        batch.pop("hist_lam_target_pixel_values")

    input_ids, labels = [], []
    for lang, action_indices, hist_indices_for_sample, use_history in zip(
        batch.pop("langs"), latent_action_idx, hist_action_idx, has_history.tolist()
    ):
        if latent_action_token_len > 0 and action_indices.numel() != latent_action_token_len:
            raise ValueError(
                "Latent action token length mismatch: "
                f"expected {latent_action_token_len}, got {action_indices.numel()}."
            )
        action_tokens = "".join(f"<ACT_{idx}>" for idx in action_indices.tolist())
        input_prompt = f"What action should the robot take to {lang}?"
        if use_history:
            hist_action_tokens = "".join(f"<ACT_{idx}>" for idx in hist_indices_for_sample.tolist())
            input_prompt += " History action " + hist_action_tokens

        prompt_builder = prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": input_prompt},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        cur_input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        cur_labels = list(cur_input_ids)
        cur_input_ids, cur_labels = torch.tensor(cur_input_ids), torch.tensor(cur_labels)
        cur_labels[: -(len(action_indices) + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            cur_labels[-1] = IGNORE_INDEX
        input_ids.append(cur_input_ids)
        labels.append(cur_labels)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
    batch["input_ids"] = input_ids
    batch["attention_mask"] = input_ids.ne(pad_token_id)
    batch["labels"] = labels
    batch["latent_action_idx"] = latent_action_idx
    return batch


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(
        f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on "
        f"{cfg.dataset_format} dataset `{cfg.dataset_name}`"
    )

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'
   
    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.config.latent_action_token_len = cfg.latent_action_token_len


    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    wrapped_model = Wrapped_Model(
        vla = vla,
        freeze_vla = cfg.freeze_vla,
        window_size=cfg.window_size,
        latent_action_token_len=cfg.latent_action_token_len,
    ).to(device_id)

    
    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)
        if cfg.use_scheduler
        else None
    )

    train_loop_lam = os.environ.get("POLAR_LAM_IN_TRAIN_LOOP", "0") == "1"
    latent_action_model = _build_latent_action_model(cfg).to(device_id).eval()
    prompt_builder_fn = PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder

    if cfg.dataset_format == "rlds":
        batch_transform = RLDSBatchTransformWindowActionWithHistory(
            None if train_loop_lam else latent_action_model,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            image_transform_lam=transforms.ToTensor(),
            prompt_builder_fn=prompt_builder_fn,
            window_size=cfg.window_size,
        )

        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(wrapped_model.module.vla.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
            window_size=cfg.window_size + 1,        # for constructing history latent actions
            training_phase='post-training',
        )
    elif cfg.dataset_format == "lerobot_cache":
        if cfg.lerobot_cache_dir is None:
            raise ValueError("--lerobot_cache_dir is required when --dataset_format lerobot_cache.")
        cache_dataset_name = None if cfg.dataset_name == "bridge" else cfg.dataset_name
        vla_dataset = LeRobotWindowCacheDataset(
            cache_dir=cfg.lerobot_cache_dir,
            image_transform=processor.image_processor.apply_transform,
            image_transform_lam=transforms.ToTensor(),
            window_size=cfg.window_size,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
            train=True,
            dataset_name=cache_dataset_name,
        )
    else:
        raise ValueError(f"Unsupported dataset_format={cfg.dataset_format!r}; expected 'rlds' or 'lerobot_cache'.")

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForWindowActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        base_tokenizer=None if train_loop_lam else processor.tokenizer,
        latent_action_model=None if train_loop_lam else latent_action_model,
        prompt_builder_fn=None if train_loop_lam else prompt_builder_fn,
        latent_action_token_len=cfg.latent_action_token_len,
        lam_token_view=cfg.lam_token_view,
    )
    dataloader_workers = int(os.environ.get("POLAR_VLA_DATALOADER_WORKERS", "0"))
    dataloader_kwargs = dict(
        dataset=vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=dataloader_workers,
    )
    if dataloader_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(os.environ.get("POLAR_VLA_DATALOADER_PREFETCH_FACTOR", "2"))
        dataloader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(**dataloader_kwargs)

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        profile_steps = int(os.environ.get("POLAR_FINETUNE_PROFILE_STEPS", os.environ.get("POLAR_PROFILE_STEPS", "0")))
        last_step_end = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader):
            profile_this_step = distributed_state.is_main_process and batch_idx < profile_steps
            data_wait_done = time.perf_counter()
            if "lam_initial_pixel_values" in batch:
                batch = _materialize_lam_tokens_in_train_loop(
                    batch,
                    latent_action_model,
                    processor.tokenizer,
                    prompt_builder_fn,
                    processor.tokenizer.model_max_length,
                    processor.tokenizer.pad_token_id,
                    cfg.latent_action_token_len,
                    cfg.lam_token_view,
                    device_id,
                )
            batch["input_ids"] = batch["input_ids"].to(device_id)
            batch["attention_mask"] = batch["attention_mask"].to(device_id)
            batch["labels"] = batch["labels"].to(device_id)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch['actions'] = batch['actions'].to(device_id)
            batch['latent_action_idx'] = batch['latent_action_idx'].to(device_id)
            if profile_this_step:
                torch.cuda.synchronize(device_id)
            h2d_done = time.perf_counter()
            
            # Forward pass
            output, act_loss, loss_one_step, latent_action_proj = wrapped_model(batch)
            loss = act_loss if cfg.freeze_vla else act_loss + output.loss
            if profile_this_step:
                torch.cuda.synchronize(device_id)
            forward_done = time.perf_counter()

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)

            # Backward pass
            normalized_loss.backward()
            if profile_this_step:
                torch.cuda.synchronize(device_id)
            backward_done = time.perf_counter()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > 32000

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()


            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

            # Push Metrics to W&B (every 5 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:
                
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "latent_action_accuracy": smoothened_action_accuracy,
                        "action_loss": act_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                progress.update()
            if profile_this_step:
                torch.cuda.synchronize(device_id)
            step_done = time.perf_counter()
            if profile_this_step:
                print(
                    "FINETUNE_PROFILE "
                    f"step={batch_idx + 1:06d} "
                    f"data_wait={data_wait_done - last_step_end:.3f}s "
                    f"h2d={h2d_done - data_wait_done:.3f}s "
                    f"forward={forward_done - h2d_done:.3f}s "
                    f"backward={backward_done - forward_done:.3f}s "
                    f"opt_log={step_done - backward_done:.3f}s "
                    f"total={step_done - last_step_end:.3f}s",
                    flush=True,
                )
                if "profile" in batch:
                    profile = batch["profile"]
                    print(
                        "FINETUNE_COLLATOR_PROFILE "
                        f"step={batch_idx + 1:06d} "
                        f"collator_total={profile['collator_total']:.3f}s "
                        f"pixel_stack={profile['collator_pixel_stack']:.3f}s "
                        f"lam={profile['collator_lam']:.3f}s "
                        f"tokenize={profile['collator_tokenize']:.3f}s "
                        f"pad={profile['collator_pad']:.3f}s",
                        flush=True,
                    )
            last_step_end = step_done

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    if not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        wrapped_model.module.vla.save_pretrained(save_dir)

                    # Save low-level policy
                    action_decoder_state = wrapped_model.module.action_decoder.state_dict()
                    torch.save(action_decoder_state, Path(run_dir) / f"action_decoder-{gradient_step_idx}.pt")
                    torch.save(action_decoder_state, Path(run_dir) / "action_decoder.pt")

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
