"""
Hybrid GPT Model with Sequential Pruning:
- Dynamic Token Reduction (RL policies) in early layers
- Learned Token Pruning (LTP) in later layers

Two-stage training:
1. Stage 1: Train LTP with unfrozen GPT2 (RL policies exist but unused)
2. Stage 2: Freeze GPT2 + LTP thresholds, train only RL policies
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks (from model.py)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Standard Causal Self-Attention (from model.py)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# LTP Causal Self-Attention (returns importance scores)
# ---------------------------------------------------------------------------

class CausalSelfAttentionLTP(nn.Module):
    """
    Causal self-attention that also produces token importance scores
    for learned token pruning.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(
            1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, C)
        attention_mask: (B, T) 1 = valid, 0 = pruned
        returns:
            y: (B, T, C)
            importance_scores: (B, T)
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)

        # causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # mask out already-pruned tokens (as keys)
        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(key_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        # Importance scores: how much attention each token receives
        importance_scores = att.mean(dim=1).mean(dim=1)  # (B, T)

        return y, importance_scores


# ---------------------------------------------------------------------------
# Standard Block (for non-pruning layers)
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# LTP Block (for LTP pruning layers)
# ---------------------------------------------------------------------------

class BlockLTP(nn.Module):
    """Transformer block with Learned Token Pruning."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionLTP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Learnable threshold for this layer
        n_ltp_layers = len(config.ltp_layers)
        ltp_layer_position = list(config.ltp_layers).index(layer_idx)
        initial_threshold = (ltp_layer_position + 1) / \
            n_ltp_layers * config.final_token_threshold
        self.threshold = nn.Parameter(torch.tensor(
            initial_threshold, dtype=torch.float32))

        self.temperature = config.temperature

    def compute_mask(self, importance_scores: torch.Tensor, masking_mode: str = "hard"):
        """Compute pruning mask from importance scores."""
        if masking_mode == "soft":
            mask = torch.sigmoid(
                (importance_scores - self.threshold) / self.temperature)
        else:
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
            protected_mask: (B, T) 1 = must never be pruned
            masking_mode: 'soft' or 'hard'
            lambda_factor: sparsity regularization weight

        Returns:
            x: (B, T, C)
            new_mask: (B, T)
            pruning_loss: scalar
        """
        attn_out, importance_scores = self.attn(self.ln_1(x), attention_mask)
        x = x + attn_out

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
        x = x * combined_mask.unsqueeze(-1)

        # MLP + residual
        x = x + self.mlp(self.ln_2(x))

        # Sparsity regularizer (only in soft mode)
        pruning_loss = torch.tensor(0.0, device=x.device)
        if masking_mode == "soft" and lambda_factor > 0.0:
            pruning_loss = lambda_factor * combined_mask.mean()

        return x, combined_mask, pruning_loss


# ---------------------------------------------------------------------------
# Token Policy (for Dynamic Token Reduction)
# ---------------------------------------------------------------------------

class TokenPolicy(nn.Module):
    """
    π(a | h) = σ(W2 * GeLU(W1 * h)), per TR-BERT paper.
    Takes token states (B, T, d_model) and outputs p(select) per token (B, T).
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        # Disable autocast and run in float32 to avoid mixed-precision numerical issues
        # This is critical for RL training where bernoulli sampling requires valid [0,1] probs
        device_type = 'cuda' if h.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_float = h.float()
            logits = self.net(h_float)
            # Clamp logits to prevent overflow in sigmoid
            logits = torch.clamp(logits, -20.0, 20.0)
            probs = torch.sigmoid(logits)
            # Clamp to valid probability range for torch.bernoulli
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        return probs.squeeze(-1)


# ---------------------------------------------------------------------------
# Hybrid Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfigHybrid:
    # Base GPT2 params
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # Dynamic Token Reduction params (early layers)
    reduction_layers: tuple = (4, 8)
    policy_hidden_dim: int = 256
    lambda_tokens: float = 1e-4
    rl_weight: float = 0.1

    # LTP params (later layers)
    ltp_layers: tuple = (9, 10, 11)
    final_token_threshold: float = 0.01
    temperature: float = 5.0
    masking_mode: str = "soft"
    lambda_factor: float = 0.1
    min_keep_tokens: int = 64


# ---------------------------------------------------------------------------
# Hybrid GPT Model
# ---------------------------------------------------------------------------

class GPTHybrid(nn.Module):
    """
    GPT-2 style language model with hybrid pruning:
    - Dynamic Token Reduction (RL policies) in early layers
    - Learned Token Pruning (LTP) in later layers
    """

    def __init__(self, config: GPTConfigHybrid):
        super().__init__()
        self.config = config

        # Build transformer blocks
        blocks = []
        for i in range(config.n_layer):
            if i in config.ltp_layers:
                blocks.append(BlockLTP(config, i))
            else:
                blocks.append(Block(config))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(blocks),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Token policies for Dynamic TR layers
        self.reduction_layers = list(config.reduction_layers)
        self.policies = nn.ModuleDict()
        for layer_idx in self.reduction_layers:
            self.policies[str(layer_idx)] = TokenPolicy(
                config.n_embd,
                config.policy_hidden_dim
            )

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_token_reduction: bool = False,
        policy_training: bool = False,
        ltp_masking_mode: Optional[str] = None,
        ltp_temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) or None
            use_token_reduction: whether to use Dynamic TR (RL policies)
            policy_training: whether to sample actions for RL training
            ltp_masking_mode: override for LTP masking mode ('soft' or 'hard')
            ltp_temperature: override for LTP temperature (higher = softer pruning)

        Returns:
            logits: (B, T, vocab_size) or (B, 1, vocab_size)
            loss: scalar or None
            rl_info: dict with policy_logprobs and num_selected_tokens (or None)
        """
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Track alive tokens (for Dynamic TR)
        alive = torch.ones(b, t, dtype=torch.bool, device=device)

        # Attention mask for LTP (float)
        attention_mask = torch.ones(b, t, dtype=torch.float32, device=device)

        # Protected mask for LTP (last min_keep_tokens never pruned)
        protected_mask = None
        if self.config.min_keep_tokens > 0:
            protected_mask = torch.zeros(
                b, t, dtype=torch.float32, device=device)
            keep = min(self.config.min_keep_tokens, t)
            protected_mask[:, -keep:] = 1.0

        # RL tracking
        policy_logprobs = []
        num_selected_tokens = []

        # LTP tracking
        total_pruning_loss = torch.tensor(0.0, device=device)

        # Use override values if provided, otherwise use config
        active_masking_mode = ltp_masking_mode if ltp_masking_mode is not None else self.config.masking_mode
        active_temperature = ltp_temperature if ltp_temperature is not None else self.config.temperature

        # Forward through transformer blocks
        for layer_idx, block in enumerate(self.transformer.h):

            if layer_idx in self.config.ltp_layers:
                # LTP block - use override parameters if provided
                # Temporarily override block's temperature if specified
                original_temp = None
                if ltp_temperature is not None:
                    original_temp = block.temperature
                    block.temperature = ltp_temperature

                x, attention_mask, pruning_loss = block(
                    x,
                    attention_mask=attention_mask,
                    protected_mask=protected_mask,
                    masking_mode=active_masking_mode,
                    lambda_factor=self.config.lambda_factor,
                )

                # Restore original temperature
                if original_temp is not None:
                    block.temperature = original_temp

                total_pruning_loss = total_pruning_loss + pruning_loss

                # Update alive mask from attention_mask
                alive = attention_mask > 0.5

            elif use_token_reduction and str(layer_idx) in self.policies:
                # Dynamic TR layer
                prev_x = x
                x_new = block(prev_x)

                policy = self.policies[str(layer_idx)]
                probs = policy(prev_x)

                # Ensure probs are valid for bernoulli (handles any remaining numerical issues)
                probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

                if policy_training and self.training:
                    # Sample Bernoulli actions
                    actions = torch.bernoulli(probs).bool()
                    actions[:, -1] = True  # Always keep last token

                    # Once dead, always dead
                    actions = alive & actions

                    # log π(a|s) with numerical stability
                    # Clamp log probs to prevent extreme gradients
                    log_probs = torch.clamp(torch.log(probs), min=-20.0)
                    log_one_minus_probs = torch.clamp(
                        torch.log(1.0 - probs), min=-20.0)
                    log_p = (
                        actions.float() * log_probs +
                        (1.0 - actions.float()) * log_one_minus_probs
                    )
                    policy_logprobs.append(log_p.sum(dim=1))
                else:
                    # Deterministic thresholding for eval
                    actions = (probs > 0.5)
                    actions[:, -1] = True
                    actions = alive & actions

                alive = actions
                num_selected_tokens.append(alive.float().sum(dim=1))

                # Freeze skipped tokens
                mask = alive.unsqueeze(-1)
                x = torch.where(mask, x_new, prev_x)

                # Update attention_mask for subsequent LTP layers
                attention_mask = alive.float()

            else:
                # Standard block
                x = block(x)

        x = self.transformer.ln_f(x)

        # Compute loss
        rl_info = None
        if targets is not None:
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

            # Add LTP pruning loss in soft mode
            if self.config.masking_mode == "soft":
                loss = lm_loss + total_pruning_loss
            else:
                loss = lm_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            lm_loss = None

        if use_token_reduction:
            rl_info = {
                "policy_logprobs": policy_logprobs,
                "num_selected_tokens": num_selected_tokens,
            }
            return logits, lm_loss if lm_loss is not None else loss, rl_info

        return logits, loss, None

    def crop_block_size(self, block_size: int):
        """Crop block size for model surgery."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block, 'attn') and hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,
                                                  :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer, respecting requires_grad."""
        param_dict = {pn: p for pn, p in self.named_parameters()
                      if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def get_pruning_stats(self, idx: torch.Tensor) -> List[Dict]:
        """Get pruning statistics for both Dynamic TR and LTP layers."""
        device = idx.device
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        alive = torch.ones(b, t, dtype=torch.bool, device=device)
        attention_mask = torch.ones(b, t, dtype=torch.float32, device=device)

        protected_mask = None
        if self.config.min_keep_tokens > 0:
            protected_mask = torch.zeros(
                b, t, dtype=torch.float32, device=device)
            keep = min(self.config.min_keep_tokens, t)
            protected_mask[:, -keep:] = 1.0

        stats = []

        for layer_idx, block in enumerate(self.transformer.h):

            if layer_idx in self.config.ltp_layers:
                x, attention_mask, _ = block(
                    x,
                    attention_mask=attention_mask,
                    protected_mask=protected_mask,
                    masking_mode="hard",
                    lambda_factor=0.0,
                )
                alive = attention_mask > 0.5
                num_kept = alive.float().sum(dim=1).mean().item()
                stats.append({
                    "layer": layer_idx,
                    "type": "LTP",
                    "avg_tokens_kept": num_kept,
                    "keep_ratio": num_kept / t,
                    "threshold": block.threshold.item(),
                })

            elif str(layer_idx) in self.policies:
                prev_x = x
                x_new = block(prev_x)

                policy = self.policies[str(layer_idx)]
                probs = policy(prev_x)

                actions = (probs > 0.5)
                actions[:, -1] = True
                actions = alive & actions
                alive = actions

                num_kept = alive.float().sum(dim=1).mean().item()
                stats.append({
                    "layer": layer_idx,
                    "type": "DynamicTR",
                    "avg_tokens_kept": num_kept,
                    "keep_ratio": num_kept / t,
                })

                mask = alive.unsqueeze(-1)
                x = torch.where(mask, x_new, prev_x)
                attention_mask = alive.float()

            else:
                x = block(x)
                num_kept = alive.float().sum(dim=1).mean().item()
                stats.append({
                    "layer": layer_idx,
                    "type": "Standard",
                    "avg_tokens_kept": num_kept,
                    "keep_ratio": num_kept / t,
                })

        return stats
