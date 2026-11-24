'''DSpark drafter aligned with the paper (arXiv 2607.05147) and its DFlash backbone.

dflash_big.py is left untouched (its checkpoint still loads there). Differences:
1. Full-prefix context (paper Eq. 2-3). Every draft layer attends to the fused target
   features of the WHOLE sequence up to the anchor, not a 4-token window. Training packs
   several anchors of one sequence into one pass; a boolean mask lets anchor a see only
   context positions <= o_a and only the rows of its own block.
2. W_K / W_V are shared between context and draft rows (Eq. 3). Fusion is W_c first,
   RMSNorm after (Eq. 2).
3. RoPE on q/k. With a long context the draft must know distances. The paper does not
   spell this out; it is my addition, the usual choice for a Qwen-style draft layer.
4. 5 layers; all 8 block rows (anchor + 7 masks) attend to each other bidirectionally.
5. Confidence head c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]]) sharing the Markov embedding W1.

Anchor convention (unchanged): o is the last position whose target features are visible,
the bonus token sits at o+1, draft slot j (1..7) predicts the token at o+1+j.
'''
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dflash_big import RMSNorm, MarkovHead, CTX_LAYERS, CTX_HS, save, load  # noqa: F401,E402

ND = 7          # draft tokens per block (gamma); training, loss and the decoding engine all use this


def rope(x, pos, theta):
    '''Rotate-half RoPE. x (B,h,S,hd), pos (B,S).'''
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = pos.float().unsqueeze(-1) * inv
    cos, sin = ang.cos().unsqueeze(1).to(x.dtype), ang.sin().unsqueeze(1).to(x.dtype)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class DSBlock(nn.Module):
    '''Pre-norm block; context and draft rows share W_K / W_V (paper Eq. 3).'''

    def __init__(self, hidden, heads, inter, theta):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(hidden), RMSNorm(hidden)
        self.q, self.k, self.v, self.o = (nn.Linear(hidden, hidden, bias=False) for _ in range(4))
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.heads, self.hd, self.theta = heads, hidden // heads, theta

    def forward(self, x, ctx, x_pos, ctx_pos, mask):
        '''x (B,M,H) block rows; ctx (B,T,H) = H_ctx (already RMSNorm-ed); mask (B,1,M,T+M) bool.'''
        B, M, H = x.shape
        h = self.norm1(x)
        sp = lambda t: t.view(B, -1, self.heads, self.hd).transpose(1, 2)
        q = rope(sp(self.q(h)), x_pos, self.theta)
        k = torch.cat([rope(sp(self.k(ctx)), ctx_pos, self.theta),
                       rope(sp(self.k(h)), x_pos, self.theta)], 2)
        v = torch.cat([sp(self.v(ctx)), sp(self.v(h))], 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, M, H))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class DSparkDrafter(nn.Module):
    def __init__(self, model, n_draft=7, layers=5, heads=8, inter=2816, markov_rank=128,
                 rope_theta=1e6, device='cuda:0', dtype=torch.float32):
        super().__init__()
        tc = getattr(model.config, 'text_config', model.config)
        self.H = tc.hidden_size
        self.n_draft = n_draft
        self.ctx_dim = len(CTX_LAYERS) * self.H
        self.device, self.dtype = device, dtype
        # frozen teacher pieces, never registered as submodules
        # Support both AutoModelForImageTextToText (has language_model wrapper)
        # and AutoModelForCausalLM (model.model is the transformer directly)
        inner = getattr(model.model, 'language_model', model.model)
        object.__setattr__(self, 'embed', inner.embed_tokens)
        object.__setattr__(self, 'lm_head', model.lm_head)
        for p in model.parameters():
            p.requires_grad_(False)
        self.ctx_fuse = nn.Linear(self.ctx_dim, self.H, bias=False)
        self.ctx_norm = RMSNorm(self.H)
        self.in_norm = RMSNorm(self.H)
        self.in_proj = nn.Linear(self.H, self.H, bias=False)
        self.anchor_emb = nn.Parameter(torch.randn(1, self.H) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(n_draft, self.H) * 0.02)
        self.blocks = nn.ModuleList([DSBlock(self.H, heads, inter, rope_theta) for _ in range(layers)])
        self.out_norm = RMSNorm(self.H)
        self.out_proj = nn.Linear(self.H, self.H, bias=False)
        self.markov = MarkovHead(tc.vocab_size, markov_rank)
        self.conf = nn.Linear(self.H + markov_rank, 1)
        self.to(device=device, dtype=dtype)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode_ctx(self, feats):
        '''feats (B,T,5H) -> H_ctx (B,T,H) = RMSNorm(W_c [H^(l1);...;H^(l5)])  (Eq. 2).
        Casts to this module's dtype: the features come from the bf16 target while the
        drafter may be fp32.'''
        return self.ctx_norm(self.ctx_fuse(feats.to(self.ctx_fuse.weight.dtype)))

    def block_mask(self, o, T):
        '''(B,1,A*M,T+A*M) bool. Row of anchor a sees context positions <= o_a and the M rows of block a.'''
        B, A = o.shape
        M = self.n_draft + 1
        dev = o.device
        ctx_vis = (torch.arange(T, device=dev).view(1, 1, 1, T) <= o.view(B, A, 1, 1))
        ctx_vis = ctx_vis.expand(B, A, M, T).reshape(B, A * M, T)
        aid = torch.arange(A, device=dev).repeat_interleave(M)
        blk_vis = (aid.view(-1, 1) == aid.view(1, -1)).unsqueeze(0).expand(B, -1, -1)
        return torch.cat([ctx_vis, blk_vis], -1).unsqueeze(1)

    def backbone(self, hctx, bonus_emb, o):
        '''hctx (B,T,H); bonus_emb (B,A,H) embedding of each bonus token; o (B,A) last visible
        context position. Returns the draft hidden states (B,A,n_draft,H).'''
        B, T, _ = hctx.shape
        A, n = o.shape[1], self.n_draft
        M = n + 1
        dt = self.in_proj.weight.dtype
        a = self.in_proj(self.in_norm(bonus_emb).to(dt)) + self.anchor_emb.to(dt)
        masks = self.mask_emb.to(dt).view(1, 1, n, -1).expand(B, A, -1, -1)
        x = torch.cat([a.unsqueeze(2), masks], 2).reshape(B, A * M, -1)
        x_pos = (o.unsqueeze(-1) + 1 + torch.arange(M, device=o.device)).reshape(B, A * M)
        ctx_pos = torch.arange(T, device=o.device).unsqueeze(0).expand(B, T)
        mask = self.block_mask(o, T)
        for blk in self.blocks:
            x = blk(x, hctx, x_pos, ctx_pos, mask)
        return self.out_proj(self.out_norm(x)).view(B, A, M, -1)[:, :, 1:]

    def logits(self, z, prev_tokens, use_markov=True):
        '''z (...,n,H); prev_tokens (...,n): token preceding each slot (slot 1 -> the bonus token).'''
        lg = self.lm_head(z.to(torch.bfloat16)).float()
        if use_markov:
            lg = lg + self.markov(prev_tokens.reshape(-1)).view(*lg.shape)
        return lg

    def confidence(self, z, prev_tokens, detach=False):
        '''Confidence logits (...,n): c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])  (Eq. 7).'''
        h = torch.cat([z, self.markov.w1(prev_tokens)], -1)
        return self.conf(h.detach() if detach else h).squeeze(-1)
