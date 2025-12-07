"""
GPT Language Model with Learned Token Pruning (LTP)
Adapted from: https://arxiv.org/pdf/2107.00910 (Learned Token Pruning for Transformers)

Key adaptations for decoder-only (causal) attention:
1. Importance scores computed from causal attention patterns
2. Progressive token pruning across layers
3. Two-stage training: soft pruning -> hard pruning
4. Position-aware thresholds for causal models
5. Attention sink protection
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# New: Position-aware threshold module
class PositionAwareThreshold(nn.Module):
    """
    Learns position-dependent thresholds for token pruning.
    Earlier positions in causal models are structurally more important,
    so they should have higher thresholds (harder to prune).
    """

    def __init__(self, max_seq_len, num_layers, num_sink_tokens=4):
        super().__init__()
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Base threshold per layer (learnable)
        self.base_threshold = nn.Parameter(torch.zeros(num_layers))

        # Position modulation: learned adjustment per position per layer
        # Shape: (max_seq_len, num_layers)
        self.position_modulation = nn.Parameter(
            torch.zeros(max_seq_len, num_layers)
        )

        # Initialize: earlier positions get positive modulation (higher threshold)
        # Later layers get higher base threshold (more pruning)
        with torch.no_grad():
            for layer_idx in range(num_layers):
                # Linear increase in base threshold across layers
                self.base_threshold[layer_idx] = (layer_idx + 1) / num_layers * 0.01

                # Position modulation: decay from early to late positions
                positions = torch.arange(max_seq_len).float()
                decay = torch.exp(-0.01 * positions)  # Exponential decay
                self.position_modulation[:, layer_idx] = decay * 0.5

    def forward(self, layer_idx, seq_len, device):
        """
        Returns threshold for each position at given layer.

        Args:
            layer_idx: which transformer layer
            seq_len: current sequence length

        Returns:
            thresholds: (seq_len,) threshold for each position
        """
        # Get position modulation for this layer
        pos_mod = self.position_modulation[:seq_len, layer_idx].to(device)

        # Combine base threshold with position modulation
        # Use sigmoid to keep thresholds in reasonable range [0, 1]
        thresholds = torch.sigmoid(self.base_threshold[layer_idx] + pos_mod)

        # Sink tokens get very high threshold (effectively never pruned)
        thresholds[:self.num_sink_tokens] = 1.0

        return thresholds

    def get_threshold_for_layer(self, layer_idx):
        """Get the base (non-position-aware) threshold for a layer"""
        return torch.sigmoid(self.base_threshold[layer_idx])


# New: Causal importance score computation module
class CausalImportanceComputer(nn.Module):
    """
    Computes token importance scores appropriate for causal (decoder-only) models.
    Multiple methods available for experimentation.
    """

    def __init__(self, method='cumulative_attention', num_sink_tokens=4):
        super().__init__()
        self.method = method
        self.num_sink_tokens = num_sink_tokens

    def forward(self, attention_weights, hidden_states=None):
        """
        Compute importance scores for each token.

        Args:
            attention_weights: (B, num_heads, T, T) attention matrix
            hidden_states: (B, T, C) hidden states (optional, for some methods)

        Returns:
            importance: (B, T) importance score for each token
        """
        if self.method == 'cumulative_attention':
            return self.cumulative_attention_importance(attention_weights)
        elif self.method == 'attention_entropy':
            return self.attention_entropy_importance(attention_weights)
        elif self.method == 'representation_similarity':
            return self.representation_similarity_importance(hidden_states)
        elif self.method == 'combined':
            return self.combined_importance(attention_weights, hidden_states)
        else:
            return self.cumulative_attention_importance(attention_weights)

    def cumulative_attention_importance(self, attention_weights):
        """
        Importance = how much attention each token receives from subsequent tokens.
        Normalized by position (earlier tokens have more potential attendees).
        """
        B, num_heads, T, _ = attention_weights.shape
        device = attention_weights.device

        # Sum attention received from all queries (column sum)
        # attention_weights[b, h, i, j] = how much query i attends to key j
        importance = attention_weights.sum(dim=2).mean(dim=1)  # (B, T)

        # Normalize by position: earlier tokens can receive attention from more positions
        # Token at position j can receive attention from positions j, j+1, ..., T-1
        # That's (T - j) positions
        position_norm = torch.arange(T, 0, -1, device=device).float()  # [T, T-1, ..., 1]
        position_norm = position_norm / T  # Normalize to [0, 1] range
        importance = importance / (position_norm.unsqueeze(0) + 1e-8)

        # Ensure sink tokens have high importance
        importance[:, :self.num_sink_tokens] = importance.max() + 1.0

        return importance

    def attention_entropy_importance(self, attention_weights):
        """
        Tokens that receive concentrated attention (low entropy) are important.
        High entropy = attention is spread out = less important as a key.
        """
        B, num_heads, T, _ = attention_weights.shape
        eps = 1e-8

        # For each key position j, look at the distribution of attention it receives
        # Transpose so we have (B, heads, key_pos, query_pos)
        attn_to_key = attention_weights.transpose(-1, -2)

        # Normalize along query dimension (who attends to this key)
        attn_sum = attn_to_key.sum(dim=-1, keepdim=True) + eps
        attn_normalized = attn_to_key / attn_sum

        # Compute entropy of attention distribution for each key
        entropy = -(attn_normalized * (attn_normalized + eps).log()).sum(dim=-1)
        entropy = entropy.mean(dim=1)  # Average over heads: (B, T)

        # Low entropy = concentrated attention = important
        # Transform entropy to importance (inverse relationship)
        importance = 1.0 / (entropy + 0.1)

        # Ensure sink tokens have high importance
        importance[:, :self.num_sink_tokens] = importance.max() + 1.0

        return importance

    def representation_similarity_importance(self, hidden_states):
        """
        Tokens similar to their neighbors are redundant (low importance).
        Unique tokens are more important.
        """
        if hidden_states is None:
            raise ValueError("hidden_states required for representation_similarity method")

        B, T, C = hidden_states.shape
        device = hidden_states.device

        if T < 2:
            return torch.ones(B, T, device=device)

        # Normalize hidden states for cosine similarity
        hidden_norm = F.normalize(hidden_states, dim=-1)

        # Similarity with previous token
        sim_prev = (hidden_norm[:, 1:] * hidden_norm[:, :-1]).sum(dim=-1)
        sim_prev = F.pad(sim_prev, (1, 0), value=0)  # First token has no previous

        # Similarity with next token
        sim_next = (hidden_norm[:, :-1] * hidden_norm[:, 1:]).sum(dim=-1)
        sim_next = F.pad(sim_next, (0, 1), value=0)  # Last token has no next

        # Average similarity
        avg_similarity = (sim_prev + sim_next) / 2

        # High similarity = redundant = low importance
        importance = 1.0 - avg_similarity

        # Ensure sink tokens have high importance
        importance[:, :self.num_sink_tokens] = importance.max() + 1.0

        return importance

    def combined_importance(self, attention_weights, hidden_states):
        """
        Combine multiple importance signals for robustness.
        """
        importance_attn = self.cumulative_attention_importance(attention_weights)

        if hidden_states is not None:
            importance_repr = self.representation_similarity_importance(hidden_states)
            # Weighted combination
            importance = 0.7 * importance_attn + 0.3 * importance_repr
        else:
            importance = importance_attn

        return importance


# New: LTP Loss module with multiple components
class LTPLoss(nn.Module):
    """
    Multi-component loss function for Learned Token Pruning.
    Balances language modeling performance with pruning objectives.
    """

    def __init__(
        self,
        lambda_sparsity=0.1,
        lambda_sink=0.01,
        lambda_monotonic=0.05,
        lambda_position=0.01,
        num_sink_tokens=4
    ):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_sink = lambda_sink
        self.lambda_monotonic = lambda_monotonic
        self.lambda_position = lambda_position
        self.num_sink_tokens = num_sink_tokens

    def forward(self, lm_loss, keep_masks, importance_scores_list):
        """
        Compute combined loss.

        Args:
            lm_loss: scalar language modeling loss
            keep_masks: list of (B, T) masks from each layer
            importance_scores_list: list of (B, T) importance scores from each layer

        Returns:
            total_loss: combined loss
            loss_dict: dictionary of individual loss components
        """
        device = lm_loss.device

        # 1. Sparsity loss: encourage pruning (penalize keeping too many tokens)
        sparsity_loss = torch.tensor(0.0, device=device)
        for mask in keep_masks:
            # Exclude sink tokens from sparsity calculation
            non_sink_mask = mask[:, self.num_sink_tokens:]
            sparsity_loss = sparsity_loss + non_sink_mask.mean()
        sparsity_loss = sparsity_loss / max(len(keep_masks), 1)

        # 2. Sink protection loss: importance of sink tokens should be high
        sink_loss = torch.tensor(0.0, device=device)
        for importance in importance_scores_list:
            if importance.shape[1] > self.num_sink_tokens:
                sink_importance = importance[:, :self.num_sink_tokens]
                # Penalize if sink importance is below 0.8
                sink_loss = sink_loss + F.relu(0.8 - sink_importance).mean()
        sink_loss = sink_loss / max(len(importance_scores_list), 1)

        # 3. Monotonic pruning loss: later layers should prune more (keep less)
        monotonic_loss = torch.tensor(0.0, device=device)
        if len(keep_masks) > 1:
            keep_ratios = [m.mean() for m in keep_masks]
            for i in range(1, len(keep_ratios)):
                # Penalize if later layer keeps MORE than earlier layer
                monotonic_loss = monotonic_loss + F.relu(keep_ratios[i] - keep_ratios[i - 1])
            monotonic_loss = monotonic_loss / (len(keep_ratios) - 1)

        # 4. Position-aware loss: later positions should be easier to prune
        position_loss = torch.tensor(0.0, device=device)
        for mask in keep_masks:
            seq_len = mask.shape[1]
            if seq_len > self.num_sink_tokens:
                positions = torch.arange(seq_len, device=device).float()
                positions = positions / seq_len  # Normalize to [0, 1]
                # We want: keep_probability * position to be low
                # i.e., later positions (high position value) should have low keep probability
                position_loss = position_loss + (mask.mean(0) * positions).mean()
        position_loss = position_loss / max(len(keep_masks), 1)

        # Combine losses
        total_loss = (
            lm_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_sink * sink_loss +
            self.lambda_monotonic * monotonic_loss +
            self.lambda_position * position_loss
        )

        loss_dict = {
            'lm_loss': lm_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sink_loss': sink_loss.item(),
            'monotonic_loss': monotonic_loss.item(),
            'position_loss': position_loss.item(),
            'total_loss': total_loss.item(),
            'avg_keep_ratio': sum(m.mean().item() for m in keep_masks) / max(len(keep_masks), 1)
        }

        return total_loss, loss_dict


class CausalSelfAttentionLTP(nn.Module):
    """
    Causal Self-Attention with Learned Token Pruning

    Key differences from standard attention:
    - Computes token importance scores from attention patterns
    - Returns attention scores for pruning decisions
    - Handles variable-length sequences due to pruning
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # Always register this buffer since we need it for importance score computation
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        # New: Importance score computer
        self.importance_computer = CausalImportanceComputer(
            method=config.importance_method,
            num_sink_tokens=config.num_sink_tokens
        )

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, T, C) input tensor
            attention_mask: (B, T) binary mask indicating which tokens are valid (1) vs pruned (0)

        Returns:
            y: (B, T, C) output tensor
            attention_weights: (B, nh, T, T) attention weights for importance computation
            importance_scores: (B, T) importance score for each token
        """
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention weights (always needed for importance scoring)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        if attention_mask is not None:
            # Mask out pruned tokens in keys
            attention_mask_expanded = attention_mask.view(B, 1, 1, T)
            att = att.masked_fill(attention_mask_expanded == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        # New: Handle NaN from all-masked rows (can happen with aggressive pruning)
        att = torch.nan_to_num(att, nan=0.0)

        att_for_output = self.attn_dropout(att)
        y = att_for_output @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # New: Compute importance scores using the importance computer
        importance_scores = self.importance_computer(att, x)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # New: Return attention weights for potential use in loss computation
        return y, att, importance_scores


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


class BlockLTP(nn.Module):
    """
    Transformer block with Learned Token Pruning
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionLTP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # New: Store config for access to sink tokens
        self.num_sink_tokens = config.num_sink_tokens

        # Learnable threshold for this layer (kept for backward compatibility)
        # Initialize with linearly increasing values across layers
        initial_threshold = (layer_idx + 1) / \
            config.n_layer * config.final_token_threshold
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

        # Temperature for soft masking (used during soft pruning stage)
        self.temperature = config.temperature

    def compute_mask(self, importance_scores, masking_mode='hard', position_thresholds=None):
        """
        Compute pruning mask based on importance scores and threshold

        Args:
            importance_scores: (B, T) importance score for each token
            masking_mode: 'soft' or 'hard'
            position_thresholds: (T,) position-aware thresholds (optional)

        Returns:
            mask: (B, T) pruning mask (0 = prune, 1 = keep)
        """
        B, T = importance_scores.shape
        device = importance_scores.device

        # New: Use position-aware thresholds if provided
        if position_thresholds is not None:
            thresholds = position_thresholds.unsqueeze(0)  # (1, T)
        else:
            thresholds = self.threshold  # scalar

        if masking_mode == 'soft':
            # Soft differentiable mask using sigmoid
            mask = torch.sigmoid(
                (importance_scores - thresholds) / self.temperature)
        else:  # hard
            # Binary mask
            mask = (importance_scores >= thresholds).float()

        # New: Always protect sink tokens (hard constraint)
        mask[:, :self.num_sink_tokens] = 1.0

        return mask

    def forward(self, x, attention_mask=None, masking_mode='hard', position_thresholds=None):
        """
        Args:
            x: (B, T, C) input tensor
            attention_mask: (B, T) binary mask indicating which tokens are valid
            masking_mode: 'soft' or 'hard' pruning
            position_thresholds: (T,) position-aware thresholds from PositionAwareThreshold module

        Returns:
            x: (B, T, C) output tensor
            new_mask: (B, T) updated attention mask after pruning
            importance_scores: (B, T) importance scores for this layer
        """
        # New: Self-attention returns attention weights and importance scores
        attn_out, attention_weights, importance_scores = self.attn(
            self.ln_1(x), attention_mask
        )
        x = x + attn_out

        # New: Compute pruning mask with position-aware thresholds
        pruning_mask = self.compute_mask(
            importance_scores,
            masking_mode,
            position_thresholds
        )

        # Combine with existing attention mask
        if attention_mask is not None:
            combined_mask = attention_mask * pruning_mask
        else:
            combined_mask = pruning_mask

        # Apply mask to token representations
        x = x * combined_mask.unsqueeze(-1)

        # MLP
        x = x + self.mlp(self.ln_2(x))

        # New: Return importance scores instead of pruning loss
        return x, combined_mask, importance_scores


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
    final_token_threshold: float = 0.01  # Threshold for the final layer
    temperature: float = 5.0  # Temperature for soft masking
    masking_mode: str = 'hard'  # 'soft' or 'hard'
    prune_mode: str = 'learned'  # 'learned', 'manual', or 'none'
    # New: Additional LTP parameters
    num_sink_tokens: int = 4  # Number of sink tokens to protect
    importance_method: str = 'cumulative_attention'  # 'cumulative_attention', 'attention_entropy', 'representation_similarity', 'combined'
    use_position_aware_threshold: bool = True  # Whether to use position-aware thresholds
    # New: Loss weights
    lambda_sparsity: float = 0.1
    lambda_sink: float = 0.01
    lambda_monotonic: float = 0.05
    lambda_position: float = 0.01


class GPTLTP(nn.Module):
    """
    GPT model with Learned Token Pruning (LTP)
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockLTP(config, i)
                            for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # New: Position-aware threshold module
        if config.use_position_aware_threshold:
            self.threshold_module = PositionAwareThreshold(
                max_seq_len=config.block_size,
                num_layers=config.n_layer,
                num_sink_tokens=config.num_sink_tokens
            )
        else:
            self.threshold_module = None

        # New: LTP loss module
        self.ltp_loss_fn = LTPLoss(
            lambda_sparsity=config.lambda_sparsity,
            lambda_sink=config.lambda_sink,
            lambda_monotonic=config.lambda_monotonic,
            lambda_position=config.lambda_position,
            num_sink_tokens=config.num_sink_tokens
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Initialize attention mask (all tokens are valid initially)
        attention_mask = torch.ones(b, t, dtype=torch.float, device=device)

        # New: Track masks and importance scores for loss computation
        all_keep_masks = []
        all_importance_scores = []

        # Forward through transformer blocks with progressive pruning
        for layer_idx, block in enumerate(self.transformer.h):
            if self.config.prune_mode == 'learned':
                # New: Get position-aware thresholds if enabled
                if self.threshold_module is not None:
                    position_thresholds = self.threshold_module(layer_idx, t, device)
                else:
                    position_thresholds = None

                x, attention_mask, importance_scores = block(
                    x,
                    attention_mask,
                    masking_mode=self.config.masking_mode,
                    position_thresholds=position_thresholds
                )

                # New: Collect for loss computation
                all_keep_masks.append(attention_mask)
                all_importance_scores.append(importance_scores)
            else:
                # No pruning - standard forward pass
                attn_out, _, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calculate language modeling loss
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # New: Use LTP loss module for combined loss
            if self.config.prune_mode == 'learned' and self.config.masking_mode == 'soft':
                loss, loss_dict = self.ltp_loss_fn(
                    lm_loss, all_keep_masks, all_importance_scores
                )
                # Store loss dict for logging (can be accessed via self.last_loss_dict)
                self.last_loss_dict = loss_dict
            else:
                loss = lm_loss
                self.last_loss_dict = {'lm_loss': lm_loss.item()}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             threshold_lr_multiplier=1.0):  # New: separate lr for thresholds
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # New: Separate parameter groups for thresholds
        threshold_params = []
        decay_params = []
        nodecay_params = []

        for name, param in param_dict.items():
            if 'threshold' in name or 'position_modulation' in name:
                threshold_params.append(param)
            elif param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
        ]

        # New: Add threshold parameters with potentially different learning rate
        if threshold_params:
            optim_groups.append({
                'params': threshold_params,
                'weight_decay': 0.0,
                'lr': learning_rate * threshold_lr_multiplier
            })

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_threshold_params = sum(p.numel() for p in threshold_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num threshold parameter tensors: {len(threshold_params)}, with {num_threshold_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        New: During generation, we use hard pruning on the prefix but don't prune newly generated tokens.
        """
        # New: Store original mode and switch to hard for generation
        original_mode = self.config.masking_mode
        self.config.masking_mode = 'hard'

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        # New: Restore original mode
        self.config.masking_mode = original_mode

        return idx

    def get_pruning_stats(self, idx):
        """
        Get pruning statistics for a given input
        Returns the number of tokens kept at each layer
        """
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        attention_mask = torch.ones(b, t, dtype=torch.float, device=device)

        stats = []
        for layer_idx, block in enumerate(self.transformer.h):
            if self.config.prune_mode == 'learned':
                # New: Get position-aware thresholds
                if self.threshold_module is not None:
                    position_thresholds = self.threshold_module(layer_idx, t, device)
                else:
                    position_thresholds = None

                x, attention_mask, importance_scores = block(
                    x,
                    attention_mask,
                    masking_mode='hard',  # Use hard masking for stats
                    position_thresholds=position_thresholds
                )
                num_kept = attention_mask.sum(dim=1).mean().item()

                # New: More detailed stats
                stats.append({
                    'layer': layer_idx,
                    'avg_tokens_kept': num_kept,
                    'keep_ratio': num_kept / t,
                    'threshold': block.threshold.item(),
                    'avg_importance': importance_scores.mean().item(),
                    'min_importance': importance_scores.min().item(),
                    'max_importance': importance_scores.max().item(),
                    # New: Per-position keep ratios
                    'position_keep_ratios': attention_mask.mean(dim=0).tolist()
                })
            else:
                attn_out, _, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        return stats

    # New: Method to visualize pruning decisions
    def visualize_pruning(self, idx, return_masks=True):
        """
        Get detailed pruning information for visualization.

        Returns:
            masks: list of (B, T) masks for each layer
            importance_scores: list of (B, T) importance scores for each layer
            thresholds: list of (T,) thresholds for each layer
        """
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        attention_mask = torch.ones(b, t, dtype=torch.float, device=device)

        masks = []
        importance_scores_list = []
        thresholds_list = []

        for layer_idx, block in enumerate(self.transformer.h):
            if self.threshold_module is not None:
                position_thresholds = self.threshold_module(layer_idx, t, device)
            else:
                position_thresholds = None

            x, attention_mask, importance_scores = block(
                x,
                attention_mask,
                masking_mode='hard',
                position_thresholds=position_thresholds
            )

            if return_masks:
                masks.append(attention_mask.detach().cpu())
                importance_scores_list.append(importance_scores.detach().cpu())
                if position_thresholds is not None:
                    thresholds_list.append(position_thresholds.detach().cpu())
                else:
                    thresholds_list.append(torch.full((t,), block.threshold.item()))

        return masks, importance_scores_list, thresholds_list


# Training helper functions
def train_ltp_soft_stage(model, train_loader, config, num_epochs=1, log_interval=100):
    """
    Stage 1: Train with soft pruning to learn thresholds

    New: Enhanced with loss logging and gradient clipping
    """
    model.config.masking_mode = 'soft'
    model.train()

    # New: Higher learning rate for thresholds during soft stage
    optimizer = model.configure_optimizers(
        weight_decay=0.0,  # No weight decay for soft stage
        learning_rate=2e-5,
        betas=(0.9, 0.999),
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        threshold_lr_multiplier=10.0  # New: Thresholds learn faster
    )

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(data, targets)
            loss.backward()

            # New: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if batch_idx % log_interval == 0:
                # New: Log detailed loss components
                loss_dict = getattr(model, 'last_loss_dict', {})
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {loss.item():.4f}")
                for key, value in loss_dict.items():
                    print(f"  {key}: {value:.4f}")

                # New: Log threshold values
                if hasattr(model, 'threshold_module') and model.threshold_module is not None:
                    print("  Layer thresholds (base):", end=" ")
                    for i in range(model.config.n_layer):
                        thresh = model.threshold_module.get_threshold_for_layer(i).item()
                        print(f"L{i}:{thresh:.4f}", end=" ")
                    print()


def train_ltp_hard_stage(model, train_loader, config, num_epochs=5, log_interval=100):
    """
    Stage 2: Fine-tune with hard pruning (thresholds fixed)

    New: Enhanced with proper threshold freezing
    """
    model.config.masking_mode = 'hard'
    model.train()

    # New: Freeze all threshold-related parameters
    for name, param in model.named_parameters():
        if 'threshold' in name or 'position_modulation' in name:
            param.requires_grad = False
            print(f"Froze parameter: {name}")

    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=1e-5,
        betas=(0.9, 0.999),
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        threshold_lr_multiplier=0.0  # Thresholds are frozen anyway
    )

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(data, targets)
            loss.backward()

            # New: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

                # New: Log pruning statistics
                stats = model.get_pruning_stats(data)
                avg_keep = sum(s['keep_ratio'] for s in stats) / len(stats)
                print(f"  Avg keep ratio: {avg_keep:.2%}")


# New: Utility function to analyze pruning behavior
def analyze_pruning_behavior(model, data_loader, num_batches=10):
    """
    Analyze pruning behavior across multiple batches.
    Useful for debugging and understanding learned thresholds.
    """
    model.eval()

    all_stats = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            stats = model.get_pruning_stats(data)
            all_stats.append(stats)

    # Aggregate statistics
    num_layers = len(all_stats[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_stats = {
            'layer': layer_idx,
            'avg_keep_ratio': sum(s[layer_idx]['keep_ratio'] for s in all_stats) / len(all_stats),
            'avg_importance_mean': sum(s[layer_idx]['avg_importance'] for s in all_stats) / len(all_stats),
            'threshold': all_stats[0][layer_idx]['threshold'],
        }
        aggregated.append(layer_stats)

    return aggregated


# New: Function to convert soft-trained model to hard inference mode
def convert_to_hard_inference(model):
    """
    Convert a soft-trained model to hard inference mode.
    This freezes thresholds and sets masking mode to hard.
    """
    model.config.masking_mode = 'hard'

    for name, param in model.named_parameters():
        if 'threshold' in name or 'position_modulation' in name:
            param.requires_grad = False

    model.eval()
    return model