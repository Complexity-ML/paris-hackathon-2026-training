"""
Training script for Token-Routed MoE with MuonTR optimizer.

10-minute time-budgeted training on a SLURM multi-node DDP cluster.
Writes `checkpoint.pt` at the end of training or on time-out.

Design choices:
    - MuonTR (per-expert Newton-Schulz orthogonalization) on 2D+ weights
    - AdamW for embeddings, norms, and 1D params
    - Per-expert adaptive LR controller (EMA-smoothed token-count ratio)
    - Linear warmup → cosine decay LR schedule
    - BF16 mixed precision via torch.amp.autocast
    - DDP across all ranks, grad accumulation for big effective batch

Self-contained — imports only torch, numpy, and model.get_model.
"""

import os
import glob
import math
import time
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import get_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir:    str = "/home/data"
    token_dtype: str = "uint16"
    seq_len:     int = 1024

    # Model — ~111M params target (dense GPT with GQA + SwiGLU)
    vocab_size:        int = 32768
    n_layer:           int = 12
    n_head:            int = 12
    n_kv_head:         int = 4
    n_embd:            int = 768
    intermediate_size: int = 2432

    # Training
    batch_size:       int   = 16
    grad_accum_steps: int   = 2
    max_lr:           float = 3e-4       # AdamW LR for embeds / norms
    muon_lr:          float = 0.02       # MuonTR LR for 2D+ weights
    min_lr_ratio:     float = 0.1        # final LR = max_lr * min_lr_ratio
    warmup_steps:     int   = 50
    max_steps:        int   = 100_000    # hard cap — time limit is the real stop
    weight_decay:     float = 0.1
    expert_weight_decay: float = 0.005
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# MuonTR optimizer — Muon with per-expert Newton-Schulz + adaptive LR
# ---------------------------------------------------------------------------

def _newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to approximate the nearest orthogonal matrix.
    Quintic polynomial with Keller Jordan's coefficients. Runs in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.bfloat16()
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.to(M.dtype)


class MuonTR(torch.optim.Optimizer):
    """
    MuonTR: Muon with per-expert Newton-Schulz for 3D expert tensors [E, H, I]
    and standard Muon for 2D weights.

    Features:
        - Per-expert NS orthogonalization (each expert slice independently)
        - Adaptive per-expert LR: target[i] = max(1, mean_count / count[i])
          EMA-smoothed over ~500 steps, clamped to [1, max_lr_ratio]
        - Expert-aware weight decay (lighter on routed experts)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        expert_lr_scale: float = 1.5,
        expert_weight_decay: float = 0.005,
        num_experts: int = 4,
        ema_decay: float = 0.998,
        max_lr_ratio: float = 4.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, expert_lr_scale=expert_lr_scale,
            expert_weight_decay=expert_weight_decay,
        )
        super().__init__(params, defaults)
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.max_lr_ratio = max_lr_ratio
        self.token_counts: Optional[torch.Tensor] = None
        self._lr_ratio_ema: Optional[torch.Tensor] = None

    def update_token_counts(self, counts: torch.Tensor):
        self.token_counts = counts

    def _compute_lr_ratio(self, device: torch.device) -> torch.Tensor:
        if self.token_counts is None:
            return torch.ones(self.num_experts, device=device)
        c = self.token_counts.float().to(device)
        mean = c.mean()
        target = (mean / c.clamp(min=1.0)).clamp(min=1.0, max=self.max_lr_ratio)
        if self._lr_ratio_ema is None or self._lr_ratio_ema.device != device:
            self._lr_ratio_ema = target.clone()
        else:
            self._lr_ratio_ema.mul_(self.ema_decay).add_(
                target, alpha=1.0 - self.ema_decay
            )
        return self._lr_ratio_ema

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ptype = group.get("param_type", "dense")
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(grad)
                buf = state["buf"]
                buf.lerp_(grad, 1 - beta)
                update = grad.lerp(buf, beta) if nesterov else buf.clone()

                # Expert LR scale applied per-group
                eff_lr = lr * (group["expert_lr_scale"] if ptype == "expert" else 1.0)
                wd = group["expert_weight_decay"] if ptype == "expert" else group["weight_decay"]

                # Per-expert ortho for [E, H, I] tensors
                if (p.dim() == 3 and p.shape[0] == self.num_experts and ptype == "expert"):
                    per_expert_lr = self._compute_lr_ratio(update.device)
                    for e in range(self.num_experts):
                        slice_upd = update[e]
                        rows, cols = slice_upd.shape
                        transposed = rows > cols
                        if transposed:
                            slice_upd = slice_upd.T
                        slice_upd = _newton_schulz(slice_upd, steps=ns_steps)
                        if transposed:
                            slice_upd = slice_upd.T
                        slice_upd *= max(1.0, rows / cols) ** 0.5
                        slice_upd *= per_expert_lr[e].item()
                        update[e] = slice_upd
                else:
                    # Standard Muon for 2D weights
                    orig_shape = update.shape
                    if update.ndim > 2:
                        update = update.view(update.shape[0], -1)
                    rows, cols = update.shape
                    transposed = rows > cols
                    if transposed:
                        update = update.T
                    update = _newton_schulz(update, steps=ns_steps)
                    if transposed:
                        update = update.T
                    update = update * (max(1.0, rows / cols) ** 0.5)
                    update = update.reshape(orig_shape)

                if wd != 0:
                    p.mul_(1 - eff_lr * wd)
                p.add_(update, alpha=-eff_lr)

        return loss


def split_params_for_muon(model: nn.Module) -> Tuple[List[dict], List[dict]]:
    """Split model parameters into (muon_groups, adamw_groups).

    Muon handles 2D+ weight matrices; AdamW handles 1D params, embeddings,
    norms, and the tied lm_head.
    """
    expert_params, shared_params, dense_params = [], [], []
    adam_decay, adam_no_decay = [], []

    adam_keywords = ("embed", "lm_head", "norm", "bias")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_adam = p.ndim < 2 or any(k in name for k in adam_keywords)
        if is_adam:
            if p.ndim < 2 or "norm" in name or "bias" in name:
                adam_no_decay.append(p)
            else:
                adam_decay.append(p)
        elif any(k in name for k in ("gate_proj_w", "up_proj_w", "down_proj_w")):
            expert_params.append(p)
        elif "shared_" in name:
            shared_params.append(p)
        else:
            dense_params.append(p)

    muon_groups = []
    if expert_params:
        muon_groups.append({"params": expert_params, "param_type": "expert"})
    if shared_params:
        muon_groups.append({"params": shared_params, "param_type": "shared"})
    if dense_params:
        muon_groups.append({"params": dense_params, "param_type": "dense"})

    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    return muon_groups, adam_groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-map all *.bin files and draw random (seq_len+1)-token windows."""

    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len = seq_len
        np_dtype = np.dtype(dtype)
        self.shards = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths = [len(s) for s in self.shards]
        self.total = sum(self.lengths)
        self.weights = [l / self.total for l in self.lengths]

    def get_batch(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(
                shard[start:start + self.seq_len + 1].astype(np.int64)
            )
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return (torch.stack(xs, dim=0).to(device, non_blocking=True),
                torch.stack(ys, dim=0).to(device, non_blocking=True))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_factor(step: int, warmup: int, max_steps: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cos


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, step: int, cfg: Config):
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "step": step,
        "model": raw.state_dict(),
        "config": asdict(cfg),
    }, cfg.checkpoint_path)
    print(f"[ckpt] saved → {cfg.checkpoint_path}  (step {step})", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="/home/data")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--vocab_size",        type=int,   default=32768)
    parser.add_argument("--n_layer",           type=int,   default=12)
    parser.add_argument("--n_head",            type=int,   default=12)
    parser.add_argument("--n_kv_head",         type=int,   default=4)
    parser.add_argument("--n_embd",            type=int,   default=768)
    parser.add_argument("--intermediate_size", type=int,   default=2432)
    parser.add_argument("--batch_size",        type=int,   default=16)
    parser.add_argument("--grad_accum_steps",  type=int,   default=2)
    parser.add_argument("--muon_lr",           type=float, default=0.02)
    parser.add_argument("--adam_lr",           type=float, default=3e-4)
    parser.add_argument("--warmup_steps",      type=int,   default=50)
    parser.add_argument("--max_steps",         type=int,   default=100_000)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enable torchao FP8 training on nn.Linear layers (B300/H100+)")
    args = parser.parse_args()

    cfg = Config(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        seq_len            = args.seq_len,
        vocab_size         = args.vocab_size,
        n_layer            = args.n_layer,
        n_head             = args.n_head,
        n_kv_head          = args.n_kv_head,
        n_embd             = args.n_embd,
        intermediate_size  = args.intermediate_size,
        batch_size         = args.batch_size,
        grad_accum_steps   = args.grad_accum_steps,
        muon_lr            = args.muon_lr,
        max_lr             = args.adam_lr,
        warmup_steps       = args.warmup_steps,
        max_steps          = args.max_steps,
        time_limit_seconds = args.time_limit_min * 60,
    )

    # ------------------------------------------------------------------ DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank, world_size, master = 0, 1, True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337 + rank)
    torch.cuda.manual_seed_all(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters (dense GPT, GQA + SwiGLU)",
              flush=True)
        print(f"[cluster] world_size={world_size}, device={device}", flush=True)

    # Optional FP8 training via torchao — must run BEFORE DDP wrap so the
    # Float8Linear modules are registered as the DDP parameters.
    if args.fp8:
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            try:
                fp8_config = Float8LinearConfig.from_recipe_name("rowwise")
            except Exception:
                fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
            # Convert only nn.Linear — skip embeddings, lm_head, and norms
            convert_to_float8_training(
                model,
                config=fp8_config,
                module_filter_fn=lambda m, fqn: isinstance(m, nn.Linear)
                    and "lm_head" not in fqn
                    and "embed" not in fqn
                    and "norm" not in fqn,
            )
            if master:
                print(f"[fp8] enabled via torchao ({type(fp8_config).__name__})",
                      flush=True)
        except ImportError:
            if master:
                print("[fp8] torchao not installed — skipping (`pip install torchao`)",
                      flush=True)
        except Exception as e:
            if master:
                print(f"[fp8] conversion failed: {e}", flush=True)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ------------------------------------------------------------------ Optimizer
    raw_model = model.module if ddp else model
    muon_groups, adam_groups = split_params_for_muon(raw_model)

    muon = MuonTR(
        muon_groups,
        lr=cfg.muon_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=cfg.weight_decay,
        expert_lr_scale=1.0,
        expert_weight_decay=cfg.weight_decay,
        num_experts=1,  # no routed experts in dense model
    )
    adam = torch.optim.AdamW(
        adam_groups,
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused="cuda" in device,
    )
    if master:
        muon_p = sum(p.numel() for g in muon_groups for p in g["params"])
        adam_p = sum(p.numel() for g in adam_groups for p in g["params"])
        print(f"[opt] MuonTR: {muon_p/1e6:.0f}M params, AdamW: {adam_p/1e6:.0f}M params",
              flush=True)

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)
    if master:
        print(f"[data] {len(dataset.shards)} shard(s), "
              f"{dataset.total:,} tokens total", flush=True)

    # ------------------------------------------------------------------ Train
    step = 0
    model.train()
    for g in muon.param_groups: g["lr_base"] = g["lr"]
    for g in adam.param_groups: g["lr_base"] = g["lr"]

    train_start = time.time()
    muon.zero_grad(set_to_none=True)
    adam.zero_grad(set_to_none=True)

    while step < cfg.max_steps:
        # Time-limit check — broadcast stop signal so all ranks agree
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.2f} min — time limit reached.",
                      flush=True)
                save_checkpoint(model, step, cfg)
            break

        # LR schedule
        factor = lr_factor(step, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
        for g in muon.param_groups:
            g["lr"] = g["lr_base"] * factor
        for g in adam.param_groups:
            g["lr"] = g["lr_base"] * factor

        step_start = time.time()
        accumulated_loss = 0.0
        for micro in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch(cfg.batch_size, device)
            sync_ctx = (model.no_sync()
                        if (ddp and micro < cfg.grad_accum_steps - 1)
                        else nullcontext())
            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        step += 1

        if master and (step <= 5 or step % 10 == 0):
            elapsed_total = time.time() - train_start
            remaining = max(0.0, cfg.time_limit_seconds - elapsed_total)
            ppl = math.exp(min(accumulated_loss, 20))
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | ppl {ppl:7.1f} | "
                f"lr×{factor:.3f} | {(time.time()-step_start)*1000:.0f}ms/step | "
                f"elapsed {elapsed_total/60:.1f}m | left {remaining/60:.1f}m",
                flush=True,
            )

    if step >= cfg.max_steps and master:
        print(f"\n[done] reached max_steps={cfg.max_steps}", flush=True)
        save_checkpoint(model, step, cfg)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
