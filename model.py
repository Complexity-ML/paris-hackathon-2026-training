"""
Dense GPT with GQA, RoPE, QK-norm, RMSNorm, Squared-ReLU GLU.

Self-contained — no external framework imports beyond torch.

Architecture:
    - Token embedding + weight-tied lm_head
    - N × Transformer block:
        RMSNorm → GQA attention (RoPE + QK-norm) → residual
        RMSNorm → Squared-ReLU GLU (gate²·up → down) → residual
    - RMSNorm → logits

Sized for ~111M params (n_embd=768, 12 layers, inter=2432).
Aligned with the MuonTR training script and the 10-minute budget.

Contract:
    get_model(config: dict) -> nn.Module
    model.forward(idx, targets=None) -> (logits, loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary positional embedding
# ---------------------------------------------------------------------------

def _build_rope_cache(dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# ---------------------------------------------------------------------------
# Grouped Query Attention with RoPE + QK norm
# ---------------------------------------------------------------------------

class GQA(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, max_seq_len: int):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head

        kv_dim = n_kv_head * self.head_dim
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)

        # QK norm — stabilizes training
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        cos, sin = _build_rope_cache(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos = self.rope_cos[:T].to(q.dtype)
        sin = self.rope_sin[:T].to(q.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        if self.n_kv_head < self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP
# ---------------------------------------------------------------------------

class SquaredReluMLP(nn.Module):
    """Two-matmul MLP with squared-ReLU activation (Primer-style).

    y = down(relu(up(x))²)

    Drops the GLU gate for a leaner forward path: only two matmuls per
    layer instead of three. To keep total MLP params comparable to a
    SwiGLU/GLU variant, `intermediate_size` should be scaled by 3/2.
    """

    def __init__(self, n_embd: int, intermediate_size: int):
        super().__init__()
        self.up_proj   = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).square())


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int,
                 intermediate_size: int, max_seq_len: int):
        super().__init__()
        self.input_norm = RMSNorm(n_embd)
        self.attn = GQA(n_embd, n_head, n_kv_head, max_seq_len)
        self.post_attn_norm = RMSNorm(n_embd)
        self.mlp = SquaredReluMLP(n_embd, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Dense GPT
# ---------------------------------------------------------------------------

class DenseGPT(nn.Module):
    """Dense transformer with U-Net style skip connections.

    Layer pairing: encoder layer i is skip-connected into decoder layer
    (n_layer - 1 - i) via a learnable scalar gate, initialized to 0.
    At init the gates are 0 → network is a plain transformer; as
    training progresses, the gates learn to route info from early to
    late layers.

    Zero-init output projections (o_proj and MLP down_proj): residual
    branches start at exactly zero so the stack is an identity at step 0.
    Muon converges faster from this init because every update is
    meaningful (no random noise to cancel).
    """

    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 n_layer: int,
                 n_head: int,
                 n_kv_head: int,
                 n_embd: int,
                 intermediate_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer

        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head, intermediate_size, seq_len)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

        # U-Net skip scalars — one per decoder layer, initialized to 0
        # (network starts as plain transformer, learns to use skips)
        self.n_skip = n_layer // 2
        self.skip_scalars = nn.Parameter(torch.zeros(self.n_skip))

        self.apply(self._init_weights)
        # Zero-init output projections (muP-like): residual branches = 0 at init
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                with torch.no_grad():
                    p.zero_()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)

        # U-Net: run first half as encoder, store outputs;
        # run second half as decoder, adding learned-scalar-gated skips
        # from the paired encoder layer.
        encoder_outputs = []
        half = self.n_skip  # = n_layer // 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < half:
                encoder_outputs.append(x)
            else:
                # Decoder layer i pairs with encoder layer (n_layer - 1 - i)
                skip_idx = self.n_layer - 1 - i
                gate = self.skip_scalars[i - half]
                x = x + gate * encoder_outputs[skip_idx]

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean",
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Competition interface
# ---------------------------------------------------------------------------

def get_model(config: dict) -> nn.Module:
    """Instantiate the dense GPT from a config dict.

    Default: ~77.5M params
        n_embd=640, 12 layers, GQA 8/2, SquaredReluMLP inter=2880.
    U-Net skip connections (6 encoder ↔ 6 decoder) + zero-init output projs.
    Sized for 10-minute training runs — smaller model → more steps → lower
    final loss in a compute-constrained regime.
    """
    return DenseGPT(
        vocab_size        = config.get("vocab_size", 32768),
        seq_len           = config.get("seq_len", 1024),
        n_layer           = config.get("n_layer", 12),
        n_head            = config.get("n_head", 8),
        n_kv_head         = config.get("n_kv_head", 2),
        n_embd            = config.get("n_embd", 640),
        intermediate_size = config.get("intermediate_size", 2880),
    )
