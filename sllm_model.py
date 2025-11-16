from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

def _rotate_half(x: Tensor) -> Tensor:
    """
    Split last dimension into pairs (even, odd) and rotate by 90 degrees.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("Rotary embedding dimension must be even.")
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """
    Generates sine and cosine projections for rotary position embeddings.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_cached_len: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        self.dim = dim
        self.base = base
        self.max_cached_len = max(0, max_cached_len)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cached_cos", torch.empty(0), persistent=False)
        self.register_buffer("cached_sin", torch.empty(0), persistent=False)
        self._cached_len: int = 0
        self._cache_dtype: Optional[torch.dtype] = None
        self._cache_device: Optional[torch.device] = None

    def warmup_cache(
        self,
        seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Pre-compute cached cos/sin tensors so that the training loop always takes the static fast-path.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be positive when warming up the rotary cache.")
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        if seq_len > self.max_cached_len:
            self.max_cached_len = seq_len
        with torch.no_grad():
            self.forward(seq_len, device=device, dtype=dtype)

    def forward(
        self, seq_len: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> tuple[Tensor, Tensor]:
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        if self.max_cached_len > 0 and seq_len <= self.max_cached_len:
            needs_update = (
                self._cached_len < seq_len
                or self._cache_dtype != dtype
                or self._cache_device != device
            )
            if needs_update:
                # compute in float32 for numerical stability, then cast
                positions = torch.arange(self.max_cached_len, device=device, dtype=torch.float32)
                freqs = torch.outer(positions, self.inv_freq.to(torch.float32))
                cos = torch.cos(freqs).to(dtype)
                sin = torch.sin(freqs).to(dtype)
                cos = torch.repeat_interleave(cos, 2, dim=-1)
                sin = torch.repeat_interleave(sin, 2, dim=-1)
                self.cached_cos = cos
                self.cached_sin = sin
                self._cached_len = self.max_cached_len
                self._cache_dtype = dtype
                self._cache_device = device
            cos = self.cached_cos[:seq_len]
            sin = self.cached_sin[:seq_len]
            return cos, sin

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(torch.float32))
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return cos, sin


def _apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int) -> tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to first rotary_dim of q and k.
    """
    cos = cos[..., :rotary_dim]
    sin = sin[..., :rotary_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot, q_pass = q.split([rotary_dim, q.size(-1) - rotary_dim], dim=-1)
    k_rot, k_pass = k.split([rotary_dim, k.size(-1) - rotary_dim], dim=-1)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat((q_rot, q_pass), dim=-1)
    k = torch.cat((k_rot, k_pass), dim=-1)
    return q, k


def _build_padding_attention_mask(
    padding_mask: Optional[Tensor],
    seq_len: int,
    device: torch.device,
) -> Optional[Tensor]:
    if padding_mask is None:
        return None

    if padding_mask.dim() != 2:
        raise ValueError("padding_mask must have shape (batch, seq_len).")
    if padding_mask.size(1) != seq_len:
        raise ValueError("padding_mask sequence length must match token sequence length.")

    pad = padding_mask.to(device=device, dtype=torch.bool)
    return pad[:, None, None, :]


def _build_causal_attention_mask(
    seq_len: int,
    device: torch.device,
) -> Tensor:
    """
    Build a causal (lower-triangular) attention mask of shape (1, 1, seq_len, seq_len),
    where True means attention is allowed and False means masked.
    """
    causal = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return causal.unsqueeze(0).unsqueeze(0)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention layer
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rotary_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_dim = rotary_dim or self.head_dim

        if self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim cannot exceed head_dim.")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even.")

        self.rotary = RotaryEmbedding(self.rotary_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: Tensor, seq_len: int, batch_size: int) -> Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask.
                If shape is (batch, seq_len), values should be 0 for masked positions.
                If shape is (batch, 1, seq_len, seq_len) or broadcastable to scores, it is applied directly.
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = [self._shape(tensor, seq_len, batch_size) for tensor in qkv]

        cos, sin = self.rotary(seq_len, device=x.device, dtype=q.dtype)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """
    Feed-forward network computing w3(silu(w1(x)) * w2(x)).
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.w1(x)
        x_2 = self.w2(x)
        gated = F.silu(x_1) * x_2
        return self.w3(gated)


class SecondOrderTransformerLayer(nn.Module):
    """
    Implements a single second-order residual step.
    Each step keeps track of position x and velocity v.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        rotary_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        step_size: float = 1.0,
    ):
        super().__init__()
        if step_size <= 0:
            raise ValueError("step_size must be positive for stable second-order updates.")
        self.attn_state_fuse = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.ffn_state_fuse = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.attn_state_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ffn_state_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.accel_norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.accel_norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            dropout=dropout,
            bias=bias,
        )
        self.feed_forward = FeedForward(embed_dim, hidden_dim=ff_hidden_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        # Learnable log step so that actual step stays positive while layers share structure.
        self.log_step = nn.Parameter(torch.log(torch.tensor(float(step_size), dtype=torch.float32)))

    def _attn_state(self, x: Tensor, v: Tensor) -> Tensor:
        fused = torch.cat((x, v), dim=-1)
        fused = self.attn_state_fuse(fused)
        return self.attn_state_norm(fused)

    def _ffn_state(self, x: Tensor, v: Tensor) -> Tensor:
        fused = torch.cat((x, v), dim=-1)
        fused = self.ffn_state_fuse(fused)
        return self.ffn_state_norm(fused)

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        layer_embedding: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Current position tensor (batch, seq_len, embed_dim)
            v: Current velocity tensor (batch, seq_len, embed_dim)
        Returns:
            Tuple of updated (x, v) after a second-order residual step.
        """
        step = torch.exp(self.log_step)

        # First update: attention predicts acceleration.
        fused = self._attn_state(x, v)
        if layer_embedding is not None:
            fused = fused + layer_embedding
        attn_in = self.accel_norm1(fused)
        accel_attn = self.attn_dropout(self.attention(attn_in, mask=mask))
        v = v + step * accel_attn
        x = x + step * v

        # Second update: feed-forward predicts acceleration.
        fused = self._ffn_state(x, v)
        if layer_embedding is not None:
            fused = fused + layer_embedding
        ff_in = self.accel_norm2(fused)
        accel_ff = self.ff_dropout(self.feed_forward(ff_in))
        v = v + step * accel_ff
        x = x + step * v

        return x, v


class SecondOrderTransformerStack(nn.Module):
    """
    Sequential stack of second-order transformer layers that integrates position/velocity states.
    """

    def __init__(
        self,
        depth: int,
        *,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        rotary_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        step_size: float = 1.0,
        initial_velocity: str = "linear",
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be a positive integer.")
        if initial_velocity not in {"zero", "linear"}:
            raise ValueError("initial_velocity must be 'zero' or 'linear'.")
        self.depth = depth
        self.layer = SecondOrderTransformerLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            rotary_dim=rotary_dim,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
            step_size=step_size,
        )
        self.initial_velocity = initial_velocity
        self.velocity_proj = nn.Linear(embed_dim, embed_dim, bias=bias) if initial_velocity == "linear" else None
        self.embed_dim = embed_dim
        self.layer_time_embed = nn.Parameter(torch.zeros(depth, embed_dim))
        self.layer_time_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def warmup_rotary_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        """
        Make sure the rotary embeddings run in cached mode once training starts.
        """
        self.layer.attention.rotary.warmup_cache(seq_len, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_velocity: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.initial_velocity == "zero" or self.velocity_proj is None:
            v = torch.zeros_like(x)
        else:
            v = self.velocity_proj(x)

        for idx in range(self.depth):
            layer_embed = self.layer_time_norm(self.layer_time_embed[idx])
            layer_embed = layer_embed.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
            x, v = self.layer(x, v, mask=mask, layer_embedding=layer_embed)

        if return_velocity:
            return x, v
        return x


class LLM(nn.Module):
    """
    Large language model wrapper around TransformerStack with tied embeddings.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        rotary_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        final_layer_norm: bool = True,
        step_size: float = 1.0,
        initial_velocity: str = "linear",
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=True)
        self.backbone = SecondOrderTransformerStack(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            rotary_dim=rotary_dim,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
            step_size=step_size,
            initial_velocity=initial_velocity,
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if final_layer_norm else None

    def warmup_rotary_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.backbone.warmup_rotary_cache(seq_len, device=device, dtype=dtype)

    def _build_attention_mask(self, padding_mask: Optional[Tensor], seq_len: int, device: torch.device) -> Tensor:
        padding = _build_padding_attention_mask(padding_mask, seq_len, device)
        causal = _build_causal_attention_mask(seq_len, device)
        if padding is None:
            return causal
        return padding & causal

    def forward(self, token_ids: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            token_ids: Tensor of shape (batch, seq_len) with token indices.
            padding_mask: Optional tensor of shape (batch, seq_len) where 1 indicates valid tokens.
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch, seq_len).")

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        x = self.token_embedding(token_ids)

        mask = self._build_attention_mask(padding_mask, seq_len, device)
        hidden = self.backbone(x, mask=mask)

        if self.final_norm is not None:
            hidden = self.final_norm(hidden)

        logits = self.output_projection(hidden)
        return logits
