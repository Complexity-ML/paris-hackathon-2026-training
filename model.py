"""
Token-Routed MoE with deterministic Zipf routing + shared expert.

Self-contained — no external framework imports beyond torch.

Architecture:
    - Embedding + weight-tied lm_head
    - N × Transformer block:
        RMSNorm → GQA attention (RoPE) → residual
        RMSNorm → Token-Routed MLP (shared SwiGLU + 4 routed experts) → residual
    - RMSNorm → logits

Routing: deterministic expert_id = token_id % num_experts (no learned router).
         Uses masked-loop dispatch (autograd-friendly, no custom Triton backward).

Contract:
    get_model(config: dict) -> nn.Module
    model.forward(idx, targets=None) -> (logits, loss)
"""

import math

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
    # q, k: [B, H, T, D]; cos, sin: [T, D]
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

        # QK norm — stabilizes training at scale
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

        # GQA: broadcast k/v to q's head count
        if self.n_kv_head < self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Token-Routed MLP with shared expert
# ---------------------------------------------------------------------------

class TokenRoutedMLP(nn.Module):
    """
    Deterministic token-routed MoE:
        expert_id = token_id % num_experts
    Each expert is a SwiGLU slice; a dense shared SwiGLU is added on top
    and processes all tokens unconditionally.

    Dispatch: masked loop over experts (autograd-friendly, no custom kernel).
    """

    def __init__(self, n_embd: int, intermediate_size: int, num_experts: int,
                 shared_intermediate_size: int, vocab_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.n_embd = n_embd
        self.expert_dim = intermediate_size // num_experts
        self.vocab_size = vocab_size

        # Routed expert weights, kept as 3D tensors for slicing speed
        self.gate_proj_w = nn.Parameter(
            torch.randn(num_experts, n_embd, self.expert_dim) * 0.02
        )
        self.up_proj_w = nn.Parameter(
            torch.randn(num_experts, n_embd, self.expert_dim) * 0.02
        )
        self.down_proj_w = nn.Parameter(
            torch.randn(num_experts, self.expert_dim, n_embd) * 0.02
        )

        # Shared expert — dense SwiGLU
        self.shared_gate = nn.Linear(n_embd, shared_intermediate_size, bias=False)
        self.shared_up = nn.Linear(n_embd, shared_intermediate_size, bias=False)
        self.shared_down = nn.Linear(shared_intermediate_size, n_embd, bias=False)

        # Deterministic token → expert mapping
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape

        # Route each token by its input id
        expert_ids = self.token_to_expert[token_ids.clamp(0, self.vocab_size - 1)]  # [B, T]
        flat_x = x.view(-1, H)
        flat_expert_ids = expert_ids.view(-1)

        # Shared expert — runs on every token
        shared_out = self.shared_down(
            F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
        )

        # Routed experts — masked loop (autograd friendly)
        routed_out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = flat_expert_ids == e
            if not mask.any():
                continue
            x_e = flat_x[mask]
            gate_e = x_e @ self.gate_proj_w[e]
            up_e = x_e @ self.up_proj_w[e]
            inter_e = F.silu(gate_e) * up_e
            routed_out[mask] = (inter_e @ self.down_proj_w[e]).to(routed_out.dtype)

        out = routed_out + shared_out
        return out.view(B, T, H)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int,
                 intermediate_size: int, num_experts: int,
                 shared_intermediate_size: int, vocab_size: int,
                 max_seq_len: int):
        super().__init__()
        self.input_norm = RMSNorm(n_embd)
        self.attn = GQA(n_embd, n_head, n_kv_head, max_seq_len)
        self.post_attn_norm = RMSNorm(n_embd)
        self.mlp = TokenRoutedMLP(
            n_embd, intermediate_size, num_experts,
            shared_intermediate_size, vocab_size,
        )

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x), token_ids)
        return x


# ---------------------------------------------------------------------------
# Token-Routed GPT
# ---------------------------------------------------------------------------

class TokenRoutedGPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 n_layer: int,
                 n_head: int,
                 n_kv_head: int,
                 n_embd: int,
                 intermediate_size: int,
                 num_experts: int,
                 shared_intermediate_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer

        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head,
                  intermediate_size, num_experts,
                  shared_intermediate_size, vocab_size, seq_len)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # Rescale output projections for deep nets
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj_w", "shared_down.weight")):
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, idx)
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
    """Instantiate the Token-Routed MoE model from a config dict.

    Default: ~111M params (n_embd=768, 12 layers, 4 routed experts + shared).
    Sized for 10-minute training runs on a multi-node cluster.
    """
    return TokenRoutedGPT(
        vocab_size               = config.get("vocab_size", 32768),
        seq_len                  = config.get("seq_len", 1024),
        n_layer                  = config.get("n_layer", 12),
        n_head                   = config.get("n_head", 12),
        n_kv_head                = config.get("n_kv_head", 4),
        n_embd                   = config.get("n_embd", 768),
        intermediate_size        = config.get("intermediate_size", 2048),
        num_experts              = config.get("num_experts", 4),
        shared_intermediate_size = config.get("shared_intermediate_size", 384),
    )
