"""DFlash drafter in the shape of the published recipe, scaled to our target.

Differences from every previous version here
--------------------------------------------
1. Markov head instead of a GRU.  Official DSpark keeps a rank-r bilinear
   bigram term:  logits_j = lm_head(z_j) + W1[x_{j-1}] @ W2^T.  It is a pure
   lookup + one matmul, captures the intra-block dependency that a parallel
   drafter loses, and costs no sequential steps.
2. hidden = 1024, i.e. the same width as the target (not 768).  A drafter that
   is narrower than its teacher cannot represent the teacher's output space.
3. FIVE context layers instead of three, spread over the stack.
4. K = 7 draft tokens per block (official block_size is 7).
5. Trained on target-generated continuations (`make_target_data.py`).

Sizing: the official Qwen3-4B drafter is ~615M params = 15.4% of its 4B
target, including a 77.8M markov head at rank 256.  Our vocabulary is 248,320
(vs 151,936), so rank 256 would cost 127M on the head alone; rank 128 brings the
head to 63.6M and the whole drafter to ~131M = 15.0% of 873M -- same ratio as
the official one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# layer indices (0-based) whose outputs feed the drafter.  Full attention sits
# at 3/7/11/15/19/23; pick five spread across the stack, as the official
# recipe does with [1, 9, 17, 25, 33].
CTX_LAYERS = (1, 7, 13, 19, 23)
# `output_hidden_states` prepends the embedding output, so hidden_states[i] is
# the output of layer i-1 and hidden_states[L] is the post-final-norm stream.
CTX_HS = tuple(i + 1 for i in CTX_LAYERS[:-1]) + (-1,)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


class DFBlock(nn.Module):
    """Draft block with dual-source KV (target context + draft tokens)."""

    def __init__(self, hidden, heads, inter, ctx_dim):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.norm2 = RMSNorm(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.k_ctx = nn.Linear(ctx_dim, hidden, bias=False)
        self.v_ctx = nn.Linear(ctx_dim, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.heads = heads
        self.hd = hidden // heads

    def forward(self, x, ctx, attn_mask):
        B, M, H = x.shape
        h = self.norm1(x)
        sp = lambda t: t.view(B, -1, self.heads, self.hd).transpose(1, 2)
        q = sp(self.q(h))
        k = torch.cat([sp(self.k_ctx(ctx)), sp(self.k(h))], 2)
        v = torch.cat([sp(self.v_ctx(ctx)), sp(self.v(h))], 2)
        m = attn_mask.view(1, 1, M, k.shape[2])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.o(a.transpose(1, 2).reshape(B, M, H))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class MarkovHead(nn.Module):
    """rank-r bigram logit bias, exactly as in the DSpark reference code:
    bias = W1[x_{j-1}] @ W2^T, added to the backbone logits."""

    def __init__(self, vocab, rank):
        super().__init__()
        self.w1 = nn.Embedding(vocab, rank)
        self.w2 = nn.Embedding(vocab, rank)
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.zeros_(self.w2.weight)

    def forward(self, prev_tokens):
        return self.w1(prev_tokens) @ self.w2.weight.T     # (B, V)


class DFlashDrafterBig(nn.Module):
    def __init__(self, model, n_draft=7, layers=4, heads=8, inter=2816,
                 ctx_window=4, markov_rank=128, device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        tc = model.config.text_config
        self.H = tc.hidden_size                      # 1024, same as the target
        self.n_draft = n_draft
        self.window = ctx_window
        self.ctx_dim = len(CTX_LAYERS) * self.H      # 5 * 1024
        self.device = device
        self.dtype = dtype
        # frozen teacher, never a submodule
        object.__setattr__(self, "embed", model.model.language_model.embed_tokens)
        object.__setattr__(self, "lm_head", model.lm_head)
        for p in model.parameters():
            p.requires_grad_(False)

        self.ctx_norm = RMSNorm(self.ctx_dim)
        self.ctx_fuse = nn.Linear(self.ctx_dim, self.H, bias=False)
        self.in_norm = RMSNorm(self.H)
        self.in_proj = nn.Linear(self.H, self.H, bias=False)
        self.anchor_emb = nn.Parameter(torch.randn(1, self.H) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(n_draft, self.H) * 0.02)
        self.blocks = nn.ModuleList(
            [DFBlock(self.H, heads, inter, self.H) for _ in range(layers)])
        self.out_norm = RMSNorm(self.H)
        self.out_proj = nn.Linear(self.H, self.H, bias=False)
        self.markov = MarkovHead(tc.vocab_size, markov_rank)
        self.to(device=device, dtype=dtype)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def block_mask(self, M, C):
        m = torch.zeros(M, C + M, dtype=torch.bool, device=self.device)
        m[:, :C] = True         # all positions see the target context
        m[1:, C:] = True        # draft positions see each other (bidirectional)
        m[0, C] = True          # anchor sees itself
        return m

    def backbone(self, ctx, anchor):
        """ctx (B,C,ctx_dim), anchor (B,1,H) bonus-token embedding.
        Returns (B, n_draft, H)."""
        B = ctx.shape[0]
        ctx = self.ctx_fuse(self.ctx_norm(ctx))
        a = self.in_proj(self.in_norm(anchor).to(self.in_proj.weight.dtype)) \
            + self.anchor_emb.to(self.in_proj.weight.dtype)
        x = torch.cat([a, self.mask_emb.to(a.dtype).unsqueeze(0).expand(B, -1, -1)], 1)
        mask = self.block_mask(x.shape[1], ctx.shape[1])
        for blk in self.blocks:
            x = blk(x, ctx, mask)
        return self.out_proj(self.out_norm(x)[:, 1:])

    def logits(self, z, prev_tokens, use_markov=True):
        """z (B,n,H); prev_tokens (B,n) holds the token preceding EACH draft
        position (for slot j that is the token at j-1)."""
        lg = self.lm_head(z.to(torch.bfloat16)).float()
        if use_markov:
            B, n, V = lg.shape
            lg = lg + self.markov(prev_tokens.reshape(-1)).view(B, n, V)
        return lg


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def precompute_onpolicy(model, tok, seqs, device="cuda:0", chunk=1024):
    """Cache the five context layers for target-generated sequences."""
    out = []
    t0 = time.perf_counter()
    for seq in seqs:
        ids = torch.tensor(seq, device=device)
        for i in range(0, max(1, ids.shape[0] - chunk), chunk):
            piece = ids[i:i + chunk]
            if piece.shape[0] < 40:
                continue
            o = model(input_ids=piece.view(1, -1), use_cache=False,
                      output_hidden_states=True)
            hs = torch.stack([o.hidden_states[j][0] for j in CTX_HS])
            out.append((hs.to(torch.bfloat16).cpu(), piece.cpu()))
    print(f"precomputed {len(out)} chunks from {len(seqs)} sequences in "
          f"{time.perf_counter()-t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(model, tok, d, chunks, steps=8000, K=8, B=64, lr=1e-3, device="cuda:0",
          log_every=100, seed=0, use_markov=True):
    teacher = {id(p) for p in model.parameters()}
    leaked = [n for n, p in d.named_parameters() if id(p) in teacher]
    if leaked:
        raise RuntimeError(f"target leaked into drafter: {leaked[:5]}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    torch.manual_seed(seed)
    d.to(device=device, dtype=torch.float32)
    d.train()
    params = [p for p in d.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    pct = 0.03 if steps >= 40 else 0.0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                               pct_start=pct)

    embed, lm_head = d.embed, d.lm_head
    W = d.window
    C = d.ctx_dim
    n_draft = K - 1
    w = torch.tensor([1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1], device=device)[:n_draft]
    w = w / w.sum()

    t0 = time.perf_counter()
    run = {"kl": 0.0, "tv": 0.0, "top1": 0.0}
    for step in range(1, steps + 1):
        ci = torch.randint(len(chunks), (B,))
        cil = ci.tolist()
        offs = [int(torch.randint(W, chunks[c][1].shape[0] - K - 2, (1,))) for c in cil]
        ctx_l, anc_l, prev_l, tgt_l, pt_l = [], [], [], [], []
        for c, o in zip(cil, offs):
            hs, idc = chunks[c]
            win = hs[:, o - W + 1:o + 1, :]                    # (5,W,H)
            ctx_l.append(win.permute(1, 0, 2).reshape(W, C))
            anc_l.append(hs[:, o, :].reshape(C))
            prev_l.append(idc[o + 1].item())
            tgt_l.append(idc[o + 2:o + 2 + n_draft].tolist())
            pt_l.append(torch.stack([hs[:, o + j, :].reshape(C)
                                     for j in range(1, K)]))
        ctx = torch.stack(ctx_l).to(device, non_blocking=True).float()
        anc = torch.stack(anc_l).to(device, non_blocking=True).float().unsqueeze(1)
        p_hidden = torch.stack(pt_l).to(device, non_blocking=True).float()
        prev_tok = torch.tensor(prev_l, device=device)
        tgt = torch.tensor(tgt_l, device=device)

        with torch.no_grad():
            last = p_hidden[:, :, -d.H:]                        # last ctx layer
            p_target = torch.softmax(lm_head(last.to(torch.bfloat16)).float(), -1)

        z = d.backbone(ctx, embed(prev_tok).float().unsqueeze(1))
        # slot 0 is conditioned on the bonus token; slot j on x_{t+j}
        prev_table = torch.cat([prev_tok.view(B, 1), tgt[:, :n_draft - 1]], 1)
        lg = d.logits(z[:, :n_draft], prev_table, use_markov)   # (B,n,V)
        q = torch.log_softmax(lg, -1)

        kl = tv = top1 = 0.0
        for j in range(n_draft):
            pj = p_target[:, j]
            kl = kl + w[j] * F.kl_div(q[:, j], pj, reduction="batchmean")
            with torch.no_grad():
                tv = tv + 0.5 * (pj - q[:, j].exp()).abs().sum(-1).mean()
                top1 = top1 + (q[:, j].argmax(-1) == pj.argmax(-1)).float().mean()

        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run["kl"] += kl.item()
        run["tv"] += (tv / n_draft).item()
        run["top1"] += (top1 / n_draft).item()

        if step % log_every == 0:
            print(f"step {step:5d}  KL {run['kl']/log_every:6.3f}  "
                  f"top1 {run['top1']/log_every:5.1%}  TV {run['tv']/log_every:5.3f}  "
                  f"({(time.perf_counter()-t0)/step*1e3:.0f} ms/step)")
            run = {"kl": 0.0, "tv": 0.0, "top1": 0.0}

    d.to(dtype=torch.bfloat16).eval()
    return d


def save(d, path):
    torch.save({k: v.detach().float().cpu() for k, v in d.state_dict().items()
                if not k.startswith(("embed.", "lm_head."))}, path)
    print("saved", path)


def load(d, path):
    sd = torch.load(path, map_location=d.device)
    missing, _ = d.load_state_dict({k: v.to(d.dtype) for k, v in sd.items()}, strict=False)
    bad = [m for m in missing if not m.startswith(("embed.", "lm_head."))]
    if bad:
        raise RuntimeError(f"missing: {bad[:6]}")
    return d


if __name__ == "__main__":
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--markov-rank", type=int, default=128)
    ap.add_argument("--data", default="/home/dev/KAP/data/dspark_legacy/onpolicy.pt")
    ap.add_argument("--out", default="/home/dev/KAP/parameter/draft/legacy/dflash_big.pt")
    a = ap.parse_args()

    P = "/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original"
    tok = AutoTokenizer.from_pretrained(P)
    model = AutoModelForImageTextToText.from_pretrained(
        P, dtype=torch.bfloat16, device_map="cuda:0").eval()

    seqs = torch.load(a.data)["seqs"]
    print(f"{len(seqs)} target-generated sequences")
    chunks = precompute_onpolicy(model, tok, seqs)
    n_tok = sum(c[1].shape[0] for c in chunks)
    print(f"{n_tok/1000:.0f}k training tokens")

    d = DFlashDrafterBig(model, n_draft=7, layers=a.layers, markov_rank=a.markov_rank)
    print(f"drafter: {d.n_params/1e6:.1f}M ({100*d.n_params/873.4e6:.1f}% of target)")
    train(model, tok, d, chunks, steps=a.steps, lr=a.lr)
    save(d, a.out)
