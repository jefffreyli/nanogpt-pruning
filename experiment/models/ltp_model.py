"""
GPT Language Model with Learned Token Pruning (LTP)
Adapted from: https://arxiv.org/pdf/2107.00910 (Learned Token Pruning for Transformers)

Key adaptations for decoder-only (causal) attention:
1. Importance scores computed from causal attention patterns
2. Progressive token pruning across layers
3. Two-stage training: soft pruning -> hard pruning
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

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, T, C) input tensor
            attention_mask: (B, T) binary mask indicating which tokens are valid (1) vs pruned (0)

        Returns:
            y: (B, T, C) output tensor
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

        # causal self-attention with importance score computation
        if self.flash:
            # Note: Flash attention doesn't return attention weights, so we need to compute them separately for pruning
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
            # Compute attention weights for importance scoring (without dropout for stability)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            if attention_mask is not None:
                # Mask out pruned tokens
                attention_mask_expanded = attention_mask.view(B, 1, 1, T)
                att = att.masked_fill(
                    attention_mask_expanded == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            if attention_mask is not None:
                # Mask out pruned tokens
                attention_mask_expanded = attention_mask.view(B, 1, 1, T)
                att = att.masked_fill(
                    attention_mask_expanded == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Compute importance scores as column mean of attention matrix
        # For causal attention, we compute how much attention each token receives from all tokens that can attend to it
        # Shape: (B, nh, T, T) -> (B, T)
        importance_scores = att.mean(dim=1)  # Average across heads: (B, T, T)
        # Column mean: how much attention does each token receive?
        # For causal attention, we only consider valid attention positions
        importance_scores = importance_scores.mean(dim=1)  # (B, T)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, importance_scores


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

        # Learnable threshold for this layer
        # Initialize with linearly increasing values across layers
        initial_threshold = (layer_idx + 1) / \
            config.n_layer * config.final_token_threshold
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

        # Temperature for soft masking (used during soft pruning stage)
        self.temperature = config.temperature

    def compute_mask(self, importance_scores, masking_mode='hard'):
        """
        Compute pruning mask based on importance scores and threshold

        Args:
            importance_scores: (B, T) importance score for each token
            masking_mode: 'soft' or 'hard'

        Returns:
            mask: (B, T) pruning mask (0 = prune, 1 = keep)
        """
        if masking_mode == 'soft':
            # Soft differentiable mask using sigmoid
            # mask = sigmoid((importance_score - threshold) / temperature)
            mask = torch.sigmoid(
                (importance_scores - self.threshold) / self.temperature)
        else:  # hard
            # Binary mask
            mask = (importance_scores >= self.threshold).float()

        return mask

    def forward(self, x, attention_mask=None, masking_mode='hard', lambda_factor=0.0):
        """
        Args:
            x: (B, T, C) input tensor
            attention_mask: (B, T) binary mask indicating which tokens are valid
            masking_mode: 'soft' or 'hard' pruning
            lambda_factor: regularization weight for encouraging pruning

        Returns:
            x: (B, T, C) output tensor
            new_mask: (B, T) updated attention mask after pruning
            pruning_loss: scalar loss for encouraging pruning (only in soft mode)
        """
        # Self-attention with importance scores
        attn_out, importance_scores = self.attn(self.ln_1(x), attention_mask)
        x = x + attn_out

        # Compute pruning mask
        pruning_mask = self.compute_mask(importance_scores, masking_mode)

        # Combine with existing attention mask
        if attention_mask is not None:
            combined_mask = attention_mask * pruning_mask
        else:
            combined_mask = pruning_mask

        # Apply mask to token representations
        if masking_mode == 'soft':
            # Soft pruning: multiply by soft mask
            x = x * combined_mask.unsqueeze(-1)
        else:
            # Hard pruning: set pruned tokens to zero
            x = x * combined_mask.unsqueeze(-1)

        # MLP
        x = x + self.mlp(self.ln_2(x))

        # Compute pruning loss (encourages more pruning)
        # Loss is the average mask value (we want to minimize this to prune more)
        pruning_loss = torch.tensor(0.0, device=x.device)
        if masking_mode == 'soft' and lambda_factor > 0:
            pruning_loss = lambda_factor * combined_mask.mean()

        return x, combined_mask, pruning_loss


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
    lambda_factor: float = 0.1  # Regularization weight for pruning loss
    prune_mode: str = 'learned'  # 'learned', 'manual', or 'none'


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

        # Track total pruning loss
        total_pruning_loss = torch.tensor(0.0, device=device)

        # Forward through transformer blocks with progressive pruning
        for block in self.transformer.h:
            if self.config.prune_mode == 'learned':
                x, attention_mask, pruning_loss = block(
                    x,
                    attention_mask,
                    masking_mode=self.config.masking_mode,
                    lambda_factor=self.config.lambda_factor
                )
                total_pruning_loss = total_pruning_loss + pruning_loss
            else:
                # No pruning - standard forward pass
                attn_out, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calculate language modeling loss
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # Add pruning loss
            if self.config.prune_mode == 'learned' and self.config.masking_mode == 'soft':
                loss = lm_loss + total_pruning_loss
            else:
                loss = lm_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
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
        for i, block in enumerate(self.transformer.h):
            if self.config.prune_mode == 'learned':
                x, attention_mask, _ = block(
                    x,
                    attention_mask,
                    masking_mode='hard',  # Use hard masking for stats
                    lambda_factor=0.0
                )
                num_kept = attention_mask.sum(dim=1).mean().item()
                stats.append({
                    'layer': i,
                    'avg_tokens_kept': num_kept,
                    'keep_ratio': num_kept / t,
                    'threshold': block.threshold.item()
                })
            else:
                attn_out, _ = block.attn(block.ln_1(x), attention_mask)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

        return stats


# Training helper functions
def train_ltp_soft_stage(model, train_loader, config, num_epochs=1):
    """
    Stage 1: Train with soft pruning to learn thresholds
    """
    model.config.masking_mode = 'soft'
    model.train()

    optimizer = model.configure_optimizers(
        weight_decay=0.0,  # No weight decay for soft stage
        learning_rate=2e-5,
        betas=(0.9, 0.999),
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(data, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")


def train_ltp_hard_stage(model, train_loader, config, num_epochs=5):
    """
    Stage 2: Fine-tune with hard pruning (thresholds fixed)
    """
    model.config.masking_mode = 'hard'
    model.train()

    # Freeze thresholds
    for block in model.transformer.h:
        block.threshold.requires_grad = False

    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=1e-5,
        betas=(0.9, 0.999),
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(data, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
