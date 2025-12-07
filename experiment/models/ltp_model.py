"""
GPT Language Model with Learned Token Pruning (LTP)

This is a GPT-2 style decoder-only transformer with:
- Learned per-layer thresholds for token pruning
- Importance scores derived from causal self-attention
- Support for "protected" tail tokens that are never pruned, so LM loss stays meaningful
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttentionLTP(nn.Module):
    """
    Causal self-attention that also produces token importance scores
    for learned token pruning.

    importance_scores: (B, T) — one scalar score per token, higher = more important.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # shape (1, 1, T, T) so it broadcasts over batch and heads
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, C)
        attention_mask: (B, T)   1 = valid, 0 = pruned
        returns:
            y: (B, T, C)
            importance_scores: (B, T)
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # shape (B, nh, T, hs)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)  # (B, nh, T, T)

        # causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # mask out already-pruned tokens (as keys)
        if attention_mask is not None:
            # (B, 1, 1, T)
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(key_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        # Importance scores: how much attention each token receives,
        # averaged over heads and query positions.
        # att: (B, nh, T_q, T_k), we aggregate over heads and queries.
        # importance[j] = mean_{h, i} att[b, h, i, j]
        importance_scores = att.mean(dim=1).mean(dim=1)  # (B, T)

        return y, importance_scores


# ---------------------------------------------------------------------------
# Block with Learned Token Pruning
# ---------------------------------------------------------------------------

class BlockLTP(nn.Module):
    """
    Transformer block with Learned Token Pruning.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionLTP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Learnable threshold for this layer
        # Initialize with linearly increasing values across layers
        initial_threshold = (layer_idx + 1) / config.n_layer * config.final_token_threshold
        self.threshold = nn.Parameter(torch.tensor(initial_threshold, dtype=torch.float32))

        # Temperature for soft masking
        self.temperature = config.temperature

    def compute_mask(self, importance_scores: torch.Tensor, masking_mode: str = "hard"):
        """
        importance_scores: (B, T)
        returns mask: (B, T), 1 = keep, 0 = prune
        """
        if masking_mode == "soft":
            # differentiable mask
            mask = torch.sigmoid((importance_scores - self.threshold) / self.temperature)
        else:
            # hard 0/1 mask
            mask = (importance_scores >= self.threshold).float()
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protected_mask: Optional[torch.Tensor] = None,
        masking_mode: str = "hard",
        lambda_factor: float = 0.0,
    ):
        """
        Args:
            x: (B, T, C)
            attention_mask: (B, T) 1 = active token, 0 = already pruned
            protected_mask: (B, T) 1 = must never be pruned (e.g. last tokens for LM loss)
            masking_mode: 'soft' or 'hard'
            lambda_factor: strength of sparsity regularizer

        Returns:
            x: (B, T, C)
            new_mask: (B, T) updated attention mask after pruning
            pruning_loss: scalar
        """
        # Attention + importance scores
        attn_out, importance_scores = self.attn(self.ln_1(x), attention_mask)
        x = x + attn_out

        # Compute pruning mask
        pruning_mask = self.compute_mask(importance_scores, masking_mode)

        # Force-keep protected tokens
        if protected_mask is not None:
            pruning_mask = torch.where(
                protected_mask > 0,
                torch.ones_like(pruning_mask),
                pruning_mask,
            )

        # Combine with existing attention mask
        if attention_mask is not None:
            combined_mask = attention_mask * pruning_mask
        else:
            combined_mask = pruning_mask

        # Apply mask to representations
        if masking_mode == "soft":
            # soft: down-weight, but not fully zero
            x = x * combined_mask.unsqueeze(-1)
        else:
            # hard: zero-out pruned tokens
            x = x * combined_mask.unsqueeze(-1)

        # MLP + residual
        x = x + self.mlp(self.ln_2(x))

        # sparsity regularizer (only in soft mode)
        pruning_loss = torch.tensor(0.0, device=x.device)
        if masking_mode == "soft" and lambda_factor > 0.0:
            pruning_loss = lambda_factor * combined_mask.mean()

        return x, combined_mask, pruning_loss


# ---------------------------------------------------------------------------
# GPT config + model wrapper
# ---------------------------------------------------------------------------

@dataclass
class GPTConfigLTP:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # LTP-specific parameters
    final_token_threshold: float = 0.01  # Global scale for thresholds
    temperature: float = 5.0             # Soft mask temperature
    masking_mode: str = "hard"           # 'soft' or 'hard'
    lambda_factor: float = 0.1           # Sparsity regularization weight
    prune_mode: str = "learned"          # 'learned' or 'none'

    # NEW: number of tail tokens that are never pruned
    # This keeps the LM loss well-posed, because we never destroy
    # the positions we still want logits for.
    min_keep_tokens: int = 64


class GPTLTP(nn.Module):
    """
    GPT-2 style language model with Learned Token Pruning.
    """

    def __init__(self, config: GPTConfigLTP):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BlockLTP(config, i) for i in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ utils

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return n_params
        # subtract embedding weights if requested
        n_embed = self.transformer.wte.weight.numel()
        return n_params - n_embed

    # ------------------------------------------------------------ main forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) token indices
        targets: (B, T) or None

        Returns:
            logits: (B, T, vocab_size)  if targets is not None
                    (B, 1, vocab_size)  if targets is None (only last position)
            loss: scalar or None
        """
        device = idx.device
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)   # (B, T, C)
        pos_emb = self.transformer.wpe(pos)   # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        # start with all tokens valid
        attention_mask = torch.ones(b, t, dtype=torch.float32, device=device)

        # protected tail: last min_keep_tokens never pruned
        if getattr(self.config, "min_keep_tokens", 0) > 0:
            protected_mask = torch.zeros_like(attention_mask)
            keep = min(self.config.min_keep_tokens, t)
            protected_mask[:, -keep:] = 1.0
        else:
            protected_mask = None

        total_pruning_loss = torch.tensor(0.0, device=device)

        # transformer blocks with progressive pruning
        for block in self.transformer.h:
            if self.config.prune_mode == "learned":
                x, attention_mask, pruning_loss = block(
                    x,
                    attention_mask,
                    protected_mask=protected_mask,
                    masking_mode=self.config.masking_mode,
                    lambda_factor=self.config.lambda_factor,
                )
                total_pruning_loss = total_pruning_loss + pruning_loss
            else:
                # vanilla block (no pruning)
                attn_out, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            if self.config.prune_mode == "learned" and self.config.masking_mode == "soft":
                loss = lm_loss + total_pruning_loss
            else:
                loss = lm_loss
        else:
            # only compute logits for last token at inference
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ------------------------------------------------------- training helpers

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # separate out weight-decay and no-decay parameters
        decay_params = []
        no_decay_params = []
        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
        return optimizer

    # ---------------------------------------------------------- pruning stats

    def get_pruning_stats(self, idx: torch.Tensor):
        """
        Run a forward pass (without computing logits) and report
        how many tokens are kept at each layer.
        """
        device = idx.device
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        attention_mask = torch.ones(b, t, dtype=torch.float32, device=device)

        # protected tail: same rule as in forward()
        if getattr(self.config, "min_keep_tokens", 0) > 0:
            protected_mask = torch.zeros_like(attention_mask)
            keep = min(self.config.min_keep_tokens, t)
            protected_mask[:, -keep:] = 1.0
        else:
            protected_mask = None

        stats = []
        for i, block in enumerate(self.transformer.h):
            if self.config.prune_mode == "learned":
                x, attention_mask, _ = block(
                    x,
                    attention_mask,
                    protected_mask=protected_mask,
                    masking_mode="hard",
                    lambda_factor=0.0,
                )
                # average tokens kept per sequence
                num_kept = attention_mask.sum(dim=1).mean().item()
                stats.append(
                    {
                        "layer": i,
                        "avg_tokens_kept": num_kept,
                        "keep_ratio": num_kept / t,
                        "threshold": block.threshold.item(),
                    }
                )
            else:
                attn_out, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        return stats
