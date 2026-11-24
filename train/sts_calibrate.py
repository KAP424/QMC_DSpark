'''Sequential Temperature Scaling (STS) for the DSpark confidence head (paper Sec. 3.2.1).

Each c_k is a conditional survival probability, so the joint prefix survival factorises into
prod_{i<=k} c_i. STS calibrates that cumulative product left to right: at position k, with the
temperatures of positions <k already fixed, a 1-D grid search picks tau_k minimising the ECE of
prod_{i<=k} sigmoid(z_i / tau_i) against the empirical cumulative acceptance prod_{i<=k} a_i.

a_i is the analytical per-step acceptance probability 1 - TV/2 = sum min(p, q), computed under
teacher forcing on the same held-out blocks.

Split: even sequence ids calibrate, odd ids evaluate. The temperatures are never chosen on the
set that reports them.
'''
from __future__ import annotations

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src4dspark'))
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_paper import DSparkDrafter, load  # noqa: E402
from dspark_loss import make_blocks, ND  # noqa: E402

W = '/home/dev/KAP/data/pb_dspark'
P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV = 'cuda:0'
A, BS = 8, 16


@torch.inference_mode()
def collect(dr, model, seqs, temp=1.0):
    '''Per-block confidence logits and analytical per-step acceptance, shape (N, 7).'''
    cs, als = [], []
    g = torch.Generator().manual_seed(0)
    for b0 in range(0, len(seqs), BS):
        feats, toks, o, p_t = make_blocks(model, tok.eos_token_id, seqs[b0:b0 + BS], A, g, temp, DEV)
        z = dr.backbone(dr.encode_ctx(feats), dr.embed(toks[..., 0]), o)
        cl = dr.confidence(z, toks[..., :ND]).float()                  # (B,A,7) logits
        lg = dr.logits(z, toks[..., :ND]) / temp
        p = p_t
        q = torch.softmax(lg, -1)
        cs.append(cl.reshape(-1, ND).cpu())
        als.append(torch.minimum(p, q).sum(-1).reshape(-1, ND).cpu())
    return torch.cat(cs), torch.cat(als)


def ece(pred, label, n_bins=15):
    edges = torch.linspace(0, 1, n_bins + 1)
    idx = torch.bucketize(pred.contiguous(), edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    total = 0.0
    n = pred.numel()
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) == 0:
            continue
        total += m.sum().item() / n * abs(pred[m].mean().item() - label[m].mean().item())
    return total


def cumulative(c, alpha):
    '''prod_{i<=k} of each column; returns (N, 7).'''
    return torch.cumprod(c, -1), torch.cumprod(alpha, -1)


def sts_fit(cl_logits, alpha, grid):
    taus = []
    cum_pred = torch.ones(cl_logits.shape[0])
    cum_lab = torch.ones(cl_logits.shape[0])
    cal_cols = []
    for k in range(ND):
        best = (float('inf'), 1.0, None)
        for t in grid:
            ck = torch.sigmoid(cl_logits[:, k] / t)
            pred = cum_pred * ck
            lab = cum_lab * alpha[:, k]
            e = ece(pred, lab)
            if e < best[0]:
                best = (e, float(t), ck)
        taus.append(best[1])
        cal_cols.append(best[2])
        cum_pred = best[2] * cum_pred
        cum_lab = cum_lab * alpha[:, k]
    return torch.tensor(taus), torch.stack(cal_cols, 1)


def report(name, cl_logits, alpha, taus=None):
    '''Two views:
    - conditional: c_k vs a_k per slot (what the head literally predicts)
    - cumulative : prod_{i<=k} c_i vs prod_{i<=k} a_i, which the scheduler consumes; E[n] is the
      SUM of the cumulative survival over slots, not its last entry.'''
    c = torch.sigmoid(cl_logits / taus) if taus is not None else torch.sigmoid(cl_logits)
    cp, cl_ = cumulative(c, alpha)
    cond = [ece(c[:, k], alpha[:, k]) for k in range(ND)]
    cum = [ece(cp[:, k], cl_[:, k]) for k in range(ND)]
    est_en = float(cp.sum(-1).mean())
    act_en = float(cl_.sum(-1).mean())
    print(f'{name:14s} ECE cond/slot ' + ' '.join(f'{x:.3f}' for x in cond) + f'   mean {sum(cond) / ND:.3f}')
    print(f'{"":14s} ECE cum /slot ' + ' '.join(f'{x:.3f}' for x in cum) + f'   mean {sum(cum) / ND:.3f}')
    print(f'{"":14s} est E[n] {est_en:.3f}  actual E[n] {act_en:.3f}  bias {est_en - act_en:+.3f}')
    return dict(cond=cond, cum=cum, est_en=est_en, act_en=act_en)


if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained(P)
    model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
    items = torch.load(W + '/prompts.pt')
    resp = {}
    for f in sorted(glob.glob(W + '/gen/shard_r*_*.pt')):
        for r in torch.load(f):
            resp[r['id']] = r
    val = [{'id': c['id'], 'prompt_ids': c['ids'], 'resp_ids': resp[c['id']]['resp']}
           for c in items if c['split'] == 'val' and c['id'] in resp and len(resp[c['id']]['resp']) >= 16]
    cal_seqs = [s for s in val if s['id'] % 2 == 0]
    eva_seqs = [s for s in val if s['id'] % 2 == 1]
    print(f'val {len(val)} -> calibrate {len(cal_seqs)}, evaluate {len(eva_seqs)}')

    dr = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
    load(dr, '/home/dev/KAP/parameter/draft/dspark_pb/dflash_pb_best.pt')
    dr.eval()

    cl_cal, al_cal = collect(dr, model, cal_seqs)
    cl_eva, al_eva = collect(dr, model, eva_seqs)
    print('blocks  calib', cl_cal.shape[0], ' eval', cl_eva.shape[0])

    grid = torch.exp(torch.linspace(torch.log(torch.tensor(0.25)), torch.log(torch.tensor(4.0)), 61))
    taus, _ = sts_fit(cl_cal, al_cal, grid)
    print('temperatures', ' '.join(f'{t:.3f}' for t in taus.tolist()))

    print(chr(10) + '--- evaluate half (temperatures never fitted here)')
    report('before (tau=1)', cl_eva, al_eva)
    report('after STS', cl_eva, al_eva, taus)
    print(chr(10) + '--- calibrate half, for reference')
    report('before (tau=1)', cl_cal, al_cal)
    report('after STS', cl_cal, al_cal, taus)

    out = {'taus': taus.tolist(), 'gamma': ND, 'n_calib': cl_cal.shape[0], 'n_eval': cl_eva.shape[0],
           'grid': [float(grid[0]), float(grid[-1]), len(grid)]}
    json.dump(out, open('/home/dev/KAP/parameter/draft/dspark_pb/sts_taus.json', 'w'), indent=2)
    print(chr(10) + 'saved /home/dev/KAP/parameter/draft/dspark_pb/sts_taus.json')
