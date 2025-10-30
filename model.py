from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
                positions = torch.arange(self.max_cached_len, device=device, dtype=dtype)
                freqs = torch.outer(positions, self.inv_freq.to(dtype))
                cos = torch.cos(freqs)
                sin = torch.sin(freqs)
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

        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(positions, self.inv_freq.to(dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
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


class TransformerLayer(nn.Module):
    """
    Transformer block with pre-norm, rotary self-attention, and gated feed-forward.
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
    ):
        super().__init__()
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            dropout=dropout,
            bias=bias,
        )
        self.feed_forward = FeedForward(embed_dim, hidden_dim=ff_hidden_dim, bias=bias)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask passed to the attention module.
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + self.attn_dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.ff_dropout(ff_out)
        return x


class TransformerStack(nn.Module):
    """
    Stack of TransformerLayer blocks applied sequentially.
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
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be a positive integer.")
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    rotary_dim=rotary_dim,
                    layer_norm_eps=layer_norm_eps,
                    bias=bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
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
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=True)
        self.backbone = TransformerStack(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            rotary_dim=rotary_dim,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps) if final_layer_norm else None

    def _build_attention_mask(self, padding_mask: Optional[Tensor], seq_len: int, device: torch.device) -> Optional[Tensor]:
        return _build_padding_attention_mask(padding_mask, seq_len, device)

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


class TimestepEmbedding(nn.Module):
    """
    Simple MLP to project scalar timesteps into the transformer embedding space.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.dim() == 2:
            timesteps = timesteps.unsqueeze(-1)
        elif timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1).unsqueeze(-1)
        if timesteps.dim() != 3:
            raise ValueError("timesteps must have shape (batch, seq_len) or (batch, seq_len, 1)")
        return self.net(timesteps)


class DLLM(nn.Module):
    """Flow-matching language model defined in logit space."""

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
        pad_token_id: int = 0,
        time_hidden_dim: Optional[int] = None,
        label_smoothing: float = 0.01,
        norm_epsilon: float = 1e-4,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.norm_epsilon = norm_epsilon

        self.token_embedding = nn.Linear(vocab_size, embed_dim)
        self.logits_mapping = nn.Linear(embed_dim, vocab_size)
        self.state_norm = nn.LayerNorm(vocab_size, eps=norm_epsilon, elementwise_affine=True)
        self.velocity_log_scale = nn.Parameter(torch.tensor(0.0))

        self.backbone = TransformerStack(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
            rotary_dim=rotary_dim,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.time_embedding = TimestepEmbedding(embed_dim, hidden_dim=time_hidden_dim)

    def _prob2logit(self, p, eps=1e-12):
        logp = torch.log(p.clamp_min(eps))
        centered = logp - logp.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1, unbiased=False, keepdim=True)
        ln_logit = centered / (std + self.norm_epsilon)
        return ln_logit
    
    def embed_token(self, input_ids, eps=1e-12):
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=self.token_embedding.weight.dtype)
        if self.label_smoothing > 0.0:
            smoothing = float(self.label_smoothing)
            off_value = smoothing / max(self.vocab_size - 1, 1)
            on_value = 1.0 - smoothing
            one_hot = one_hot * on_value + (1.0 - one_hot) * off_value
        logits = self._prob2logit(one_hot, eps)
        return self.state_norm(logits)

    def _sample_logits(self, shape: tuple[int, ...], device=None, dtype=None):
        device = device or self.token_embedding.weight.device
        dtype = dtype or self.token_embedding.weight.dtype
        x = torch.randn(*shape, device=device, dtype=dtype)
        return self.state_norm(x)

    def _expand_timesteps(self, timesteps: Optional[Tensor], batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        if timesteps is None:
            return torch.rand(batch_size, seq_len, 1, device=device)

        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1)
        elif timesteps.dim() == 2:
            if timesteps.size(0) != batch_size:
                raise ValueError("timesteps batch dimension must match token_ids batch size.")
            if timesteps.size(1) == 1:
                timesteps = timesteps.expand(batch_size, seq_len)
            timesteps = timesteps.unsqueeze(-1)
        elif timesteps.dim() == 3:
            if timesteps.size(0) != batch_size or timesteps.size(1) != seq_len or timesteps.size(2) != 1:
                raise ValueError("timesteps must have shape (batch, seq_len, 1).")
        else:
            raise ValueError("timesteps must be scalar, (batch,), (batch, seq_len), or (batch, seq_len, 1).")

        return timesteps.to(device=device, dtype=torch.float32)

    def _logits_to_embedding(self, logits: Tensor) -> Tensor:
        return self.token_embedding(logits)

    def forward(
        self,
        logits: Tensor,
        timesteps: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.predict_velocity(logits, timesteps=timesteps, padding_mask=padding_mask)

    def predict_velocity(
        self,
        logits: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if logits.dim() != 3 or logits.size(-1) != self.vocab_size:
            raise ValueError("logits must have shape (batch, seq_len, vocab_size).")
        batch_size, seq_len, _ = logits.shape
        logits = self.state_norm(logits)
        device = logits.device
        timesteps = self._expand_timesteps(timesteps, batch_size, seq_len, device)
        inputs = self._logits_to_embedding(logits)
        inputs = inputs + self.time_embedding(timesteps)
        attn_mask = _build_padding_attention_mask(padding_mask, seq_len, device)
        hidden = self.backbone(inputs, mask=attn_mask)
        hidden = self.final_norm(hidden)
        velocity = self.logits_mapping(hidden)
        scale = torch.exp(self.velocity_log_scale.clamp(-4.0, 4.0)).to(velocity.dtype)
        velocity = velocity * scale
        return velocity

    def loss(
        self,
        token_ids: Tensor,
        *,
        timesteps: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch, seq_len).")

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        target_logits = self.embed_token(token_ids)

        if timesteps is None:
            t = torch.rand(batch_size, seq_len, 1, device=device, dtype=target_logits.dtype)
        else:
            t = self._expand_timesteps(timesteps, batch_size, seq_len, device).to(target_logits.dtype)

        init_logits = self._sample_logits(target_logits.shape, device=device, dtype=target_logits.dtype)

        state_logits = (1.0 - t) * init_logits + t * target_logits
        velocity_target = target_logits - init_logits

        if padding_mask is not None:
            mask = padding_mask.to(device=device, dtype=target_logits.dtype).unsqueeze(-1)
            target_logits = target_logits * mask
            init_logits = init_logits * mask
            state_logits = state_logits * mask
            velocity_target = velocity_target * mask

        velocity_pred = self.predict_velocity(state_logits, timesteps=t, padding_mask=padding_mask)

        velocity_mse = (velocity_pred - velocity_target) ** 2

        weights = torch.ones_like(t)
        if padding_mask is not None:
            weights = weights * padding_mask.unsqueeze(-1)

        weighted_error = velocity_mse * weights
        denom = weights.sum().clamp_min(1.0)
        flow_loss = weighted_error.sum() / denom
        flow_loss = flow_loss / math.sqrt(self.vocab_size)

        #cos_loss = 1.0 - (velocity_pred * velocity_target).sum(dim=-1) / (
        #    velocity_pred.norm(dim=-1) * velocity_target.norm(dim=-1) + 1e-12
        #)
        #weighted_cos_loss = (cos_loss * weights.squeeze(-1)).sum() / weights.sum().clamp_min(1.0)
        loss = flow_loss

        if return_components:
            components = {
                "loss_total": loss.detach(),
                "loss_flow": flow_loss.detach(),
                "loss_velocity_mse": velocity_mse.mean().detach(),
            }
            return loss, components

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        steps: int = 20,
        temperature: float = 1.0,
        deterministic: bool = True,
        padding_mask: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        conditional_tokens: Optional[Tensor] = None,
        sampler: str = "heun",
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        device = device or self.token_embedding.weight.device
        logits = self._sample_logits((batch_size, seq_len, self.vocab_size), device=device)

        if padding_mask is None:
            padding_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        else:
            padding_mask = padding_mask.to(device=device)

        padding_float = padding_mask.to(dtype=logits.dtype).unsqueeze(-1)
        logits = logits * padding_float

        conditional_mask_bool = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        conditional_logits = None
        if conditional_tokens is not None:
            if conditional_tokens.dim() == 1:
                conditional_tokens = conditional_tokens.unsqueeze(0).expand(batch_size, -1)
            if conditional_tokens.shape != (batch_size, seq_len):
                raise ValueError("conditional_tokens must have shape (batch_size, seq_len)")
            conditional_mask_bool = conditional_tokens != self.pad_token_id
            if conditional_mask_bool.any():
                conditional_logits = self.embed_token(conditional_tokens)
                mask_float = conditional_mask_bool.unsqueeze(-1).to(dtype=logits.dtype)
                logits = logits * (1.0 - mask_float) + conditional_logits * mask_float

        conditional_mask_float = conditional_mask_bool.unsqueeze(-1).to(dtype=logits.dtype)

        t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)
        update_mask = padding_float * (1.0 - conditional_mask_float)

        def build_t(value: float) -> Tensor:
            t_tensor = torch.full((batch_size, seq_len, 1), value, device=device, dtype=logits.dtype)
            if conditional_mask_bool.any():
                t_tensor = t_tensor.masked_fill(conditional_mask_bool.unsqueeze(-1), 1.0)
            return t_tensor

        sampler_name = sampler.lower()
        if sampler_name == "euler":
            for idx in range(steps):
                t_curr = t_schedule[idx]
                t_next = t_schedule[idx + 1]
                delta = t_next - t_curr
                t_tensor = build_t(float(t_curr))

                velocity = self.predict_velocity(logits, timesteps=t_tensor, padding_mask=padding_mask)
                logits = logits + delta * velocity * update_mask

                if conditional_logits is not None:
                    logits = torch.where(conditional_mask_bool.unsqueeze(-1), conditional_logits, logits)
        elif sampler_name == "heun":
            for idx in range(steps):
                t_curr = t_schedule[idx]
                t_next = t_schedule[idx + 1]
                delta = t_next - t_curr

                t_tensor_curr = build_t(float(t_curr))
                velocity_curr = self.predict_velocity(logits, timesteps=t_tensor_curr, padding_mask=padding_mask)

                logits_tilde = logits + delta * velocity_curr * update_mask
                if conditional_logits is not None:
                    logits_tilde = torch.where(conditional_mask_bool.unsqueeze(-1), conditional_logits, logits_tilde)

                t_tensor_next = build_t(float(t_next))
                velocity_next = self.predict_velocity(logits_tilde, timesteps=t_tensor_next, padding_mask=padding_mask)

                logits = logits + delta * 0.5 * (velocity_curr + velocity_next) * update_mask

                if conditional_logits is not None:
                    logits = torch.where(conditional_mask_bool.unsqueeze(-1), conditional_logits, logits)
        else:
            raise ValueError(f"Unknown sampler '{sampler}'. Supported samplers: 'euler', 'heun'.")

        logits = self.state_norm(logits)
        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            tokens = probs.argmax(dim=-1)
        else:
            tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)

        if conditional_tokens is not None and conditional_mask_bool.any():
            tokens = torch.where(conditional_mask_bool, conditional_tokens, tokens)

        return tokens
