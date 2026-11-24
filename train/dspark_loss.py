'''Block sampling, the paper's loss, and evaluation for DSparkDrafter.

L = sum_k w_k (0.1*CE_k + 0.9*TV_k + 1.0*BCE_k),   w_k = exp(-(k-1)/gamma)   (paper Eq. 9-12)
  CE_k  = -log q_k(x*_k)       x*_k: the token the target actually sampled
  TV_k  = || q_k - p_k ||_1    p: target distribution, q: draft distribution, both at temperature T
  BCE_k : confidence head against c*_k = 1 - TV_k/2 (= sum min(p,q)); c* is a label, no gradient
Summed over positions (not normalised), averaged over blocks.
'''
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src4dspark'))
from dflash_big import CTX_HS  # noqa: E402
from dspark_paper import ND  # noqa: E402,F401  (gamma = 7, defined once in src4dspark)
K = ND + 1                  # bonus token + gamma draft tokens
W_CE, W_TV, W_CONF = 0.1, 0.9, 1.0


def pos_weights(device, gamma=ND):
    return torch.exp(-torch.arange(gamma, device=device, dtype=torch.float32) / gamma)


@torch.no_grad()
def make_blocks(model, pad_id, seqs, A, gen, temp, dev='cuda:0'):
    '''Draw A random anchors inside each response.
    Returns feats (B,T,5H) bf16 target features of the whole sequences, toks (B,A,8) = bonus + 7
    sampled targets, o (B,A) last visible context position, p_t (B,A,7,V) target dist at temp.'''
    full = [s['prompt_ids'] + s['resp_ids'] for s in seqs]
    # filter out sequences too short to fit even one anchor (need len(full) - K > len(prompt) - 1)
    valid = [(s, f) for s, f in zip(seqs, full) if len(f) - K > len(s['prompt_ids']) - 1]
    if not valid:
        return None, None, None, None
    seqs, full = zip(*valid)
    B, T = len(full), max(map(len, full))
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    for i, x in enumerate(full):
        ids[i, :len(x)] = torch.tensor(x)           # right padding: causal model, earlier positions unaffected
    ids = ids.to(dev)
    hs = model(input_ids=ids, use_cache=False, output_hidden_states=True, logits_to_keep=1).hidden_states
    feats = torch.stack([hs[j] for j in CTX_HS], 2).flatten(2)
    final = hs[-1]                                    # post final-norm stream = lm_head input
    # o in [P-1, L-K-1]: bonus at o+1 is the first response token at the earliest, o+8 stays in the sequence
    o = torch.stack([torch.randint(len(s['prompt_ids']) - 1, len(f) - K, (A,), generator=gen)
                     for s, f in zip(seqs, full)]).to(dev)
    b = torch.arange(B, device=dev).view(B, 1, 1)
    toks = ids[b, o.unsqueeze(-1) + torch.arange(1, K + 1, device=dev)]
    hid = final[b, o.unsqueeze(-1) + torch.arange(1, ND + 1, device=dev)]
    p_t = torch.softmax(model.lm_head(hid).float() / temp, -1)
    return feats, toks, o, p_t


def dspark_loss(dr, feats, toks, o, p_t, temp, use_markov=True, conf_detach=False):
    hctx = dr.encode_ctx(feats.float())
    z = dr.backbone(hctx, dr.embed(toks[..., 0]).float(), o)         # (B,A,7,H)
    prev = toks[..., :ND]                                              # slot j is conditioned on token j-1
    lg = dr.logits(z, prev, use_markov) / temp
    cl = dr.confidence(z, prev, detach=conf_detach)                    # (B,A,7) logits
    logq = torch.log_softmax(lg, -1)
    ce = -logq.gather(-1, toks[..., 1:].unsqueeze(-1)).squeeze(-1)    # (B,A,7)
    tv = (logq.exp() - p_t).abs().sum(-1)                              # (B,A,7)
    alpha = 1.0 - 0.5 * tv.detach()                                    # = sum min(p,q), acceptance prob
    bce = F.binary_cross_entropy_with_logits(cl, alpha, reduction='none')
    per = W_CE * ce + W_TV * tv + W_CONF * bce
    loss = (per * pos_weights(per.device)).sum(-1).mean()
    with torch.no_grad():
        m = lambda t: t.float().mean((0, 1))
        st = dict(ce=m(ce), tv=m(tv), bce=m(bce), alpha=m(alpha), conf=m(torch.sigmoid(cl)),
                  hit=m(logq.argmax(-1) == p_t.argmax(-1)))
    return loss, st


def exp_len(x):
    '''Sum_k prod_{j<=k} x_j over per-slot acceptance estimates x (7,): expected accepted drafts.'''
    return torch.cumprod(x, 0).sum().item()


def summarize(st):
    '''tau counts the bonus token (paper footnote 4): tau = E[n] + 1.'''
    return dict(tau=exp_len(st['alpha']) + 1.0, tau_argmax=exp_len(st['hit']) + 1.0,
                ce=st['ce'].mean().item(), tv=st['tv'].mean().item(), bce=st['bce'].mean().item())


@torch.no_grad()
def evaluate(dr, model, pad_id, seqs, temp, A=8, bs=16, seed=123, dev='cuda:0'):
    was = dr.training
    dr.eval()
    g = torch.Generator().manual_seed(seed)
    acc, n = None, 0
    for b0 in range(0, len(seqs), bs):
        feats, toks, o, p_t = make_blocks(model, pad_id, seqs[b0:b0 + bs], A, g, temp, dev)
        _, st = dspark_loss(dr, feats, toks, o, p_t, temp)
        acc = st if acc is None else {k: acc[k] + st[k] for k in st}
        n += 1
    dr.train(was)
    return {k: v / n for k, v in acc.items()}
