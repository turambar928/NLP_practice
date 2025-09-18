# -*- coding: utf-8 -*-
# from_scratch_pytorch_skeleton.py
# Minimal, didactic PyTorch skeletons for CNN / RNN / Transformer implementations
# matching the math in the companion lecture. Use for learning or as a base to extend.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ Utilities ============
def causal_mask(sz: int, device=None, dtype=torch.bool):
    """Return a [sz, sz] causal mask with True for invalid (masked) positions."""
    i = torch.arange(sz, device=device)
    j = torch.arange(sz, device=device)
    return (j[None, :] > i[:, None]).to(dtype)


def seq_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    """
    Cross-entropy for sequence tasks.
    logits: [B, T, V], targets: [B, T]
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=ignore_index)
    return loss


# ============ CNN ============
class SimpleConvBlock(nn.Module):
    """
    Conv2d (cross-correlation) + BN + ReLU with explicit shapes.
    """
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return y


# ============ RNN (Vanilla) ============
class VanillaRNNCell(nn.Module):
    """
    A single RNN cell with tanh activation:
      a_t = W_xh x_t + W_hh h_{t-1} + b_h
      h_t = tanh(a_t)
    """
    def __init__(self, d_x: int, d_h: int):
        super().__init__()
        self.W_xh = nn.Linear(d_x, d_h, bias=True)
        self.W_hh = nn.Linear(d_h, d_h, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        a_t = self.W_xh(x_t) + self.W_hh(h_prev)
        h_t = torch.tanh(a_t)
        return h_t


class SimpleRNN(nn.Module):
    def __init__(self, d_x: int, d_h: int, d_out: int):
        super().__init__()
        self.cell = VanillaRNNCell(d_x, d_h)
        self.readout = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d_x]
        return logits [B, T, d_out], last hidden [B, d_h]
        """
        B, T, d_x = x.shape
        d_h = self.cell.W_hh.out_features
        if h0 is None:
            h = x.new_zeros(B, d_h)
        else:
            h = h0
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(self.readout(h).unsqueeze(1))
        logits = torch.cat(outs, dim=1)
        return logits, h


# ============ Transformer ============
class LayerNorm(nn.Module):
    """Pre-LN style LayerNorm with eps."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias


class ScaledDotProductAttention(nn.Module):
    """
    Single-head attention to mirror the math.
    """
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [B, T, d_model]
        mask: [B, 1, T, T] or [B, T, T] with True for masked positions
        """
        Q = self.W_Q(H)  # [B, T, d_k]
        K = self.W_K(H)  # [B, T, d_k]
        V = self.W_V(H)  # [B, T, d_k]

        # scores
        S = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, T, T]

        if mask is not None:
            # mask=True means invalid; set to -inf
            S = S.masked_fill(mask, float('-inf'))

        A = F.softmax(S, dim=-1)  # [B, T, T]
        U = torch.matmul(A, V)    # [B, T, d_k]
        return U, A
    



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = H.shape
        # project
        Q = self.q_proj(H).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,T,d_k]
        K = self.k_proj(H).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(H).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        S = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,h,T,T]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B,1,T,T]
            S = S.masked_fill(mask, float('-inf'))
        A = F.softmax(S, dim=-1)
        U = torch.matmul(A, V)  # [B,h,T,d_k]

        U = U.transpose(1, 2).contiguous().view(B, T, D)  # concat heads
        O = self.o_proj(U)
        return O, A


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act=F.gelu):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        y, att = self.mha(self.ln1(x), mask=mask)
        x = x + self.drop1(y)
        z = self.ffn(self.ln2(x))
        x = x + self.drop2(z)
        return x, att


class TinyTransformerLM(nn.Module):
    """
    Minimal causal LM to validate equations quickly.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4, d_ff: int = 1024, num_layers: int = 4, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, T = idx.shape
        assert T <= self.max_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]

        mask = causal_mask(T, device=idx.device).unsqueeze(0)  # [1,T,T]
        attn_maps = []
        for blk in self.blocks:
            x, att = blk(x, mask=mask)
            attn_maps.append(att)
        x = self.ln_f(x)
        logits = self.head(x)  # [B,T,V]
        return logits, tuple(attn_maps)


# ============ Training Skeletons ============
def train_step_transformer(model: nn.Module, batch_idx: torch.Tensor, batch_tgt: torch.Tensor, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(batch_idx)  # [B,T,V]
    loss = seq_cross_entropy(logits, batch_tgt, ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def tiny_demo():
    """
    Quick run to verify shapes & a forward/backward pass.
    """
    torch.manual_seed(0)
    V = 100
    B, T = 2, 16
    model = TinyTransformerLM(vocab_size=V, d_model=128, num_heads=4, d_ff=256, num_layers=2, max_len=64)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss0 = train_step_transformer(model, x, y, opt)
    print("Initial loss:", loss0)


if __name__ == "__main__":
    tiny_demo()
