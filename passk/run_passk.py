"""Sampling settings benchmark across Maze, Sudoku, Countdown, and 1D-ARC.

Fixes vs previous run_settings.py / run_settings_domain.py:
  1. DSpark arithmetic-coding setting (S5_dspark_ac): single sequence-level uniform u_b
     with target-CDF exact commit (Phi_p(u_b)), achieving paper-level pass@k AND DSpark
     speculative speedup (tau > 1).
  2. Correct estimators per QuasiMoTTo (arXiv:2607.01179, Section 2.4.1 & Theorem 1):
     - S1 (i.i.d.): unbiased Chen et al. estimator 1 - C(N-c, k)/C(N, k) for E[1-(1-p)^k],
       plus analytic 1-(1-p)^k curve plotted alongside min(kp, 1).
     - Coupled lattice settings (S2-S6): Theorem-1 dyadic stride bootstrap
       (any-hit over stride N//k subsequences), NOT the i.i.d. combinatoric formula.
  3. Exact upper bound min(kp, 1) (Theorem 2):
     - Maze / Sudoku: teacher-forced exact sequence probability INCLUDING EOS token
       pi_theta(answer + <eos> | prompt); sufficient sample size (default 100 prompts)
       eliminates small-sample Monte Carlo overshoot.
     - Countdown: multiple valid expressions exist per problem, so p_i is estimated via
       a high-precision independent Monte Carlo rollout (M=128) as in QuasiMoTTo eval.py --ub.
     - 1D-ARC: uses the base model (Qwen3.5-0.8B-Base-original) + non-leaked in-context
       prompt format from QuasiMoTTo tasks.py (with ' ' + answer boundary token for exact
       teacher-forced p_i), avoiding the 901-task train/val memorization collapse.
  4. Missing final/EOS anchor commit bug in SettingEngine & DSparkDecodeFast fixed.

Settings compared:
  S1_iid           : DSpark classic accept, all streams (draft u, accept v, bonus w) i.i.d.
  S2_draft         : DSpark classic accept, coupled draft u (lattice), accept/bonus i.i.d.
  S3_draft_acc     : DSpark classic accept, coupled draft u + accept v, bonus i.i.d.
  S4_draft_acc_bns : DSpark classic accept, coupled draft u + accept v + bonus w
  S5_dspark_ac     : DSpark arithmetic-coding (single sequence uniform u, exact Phi_p(u))
  S6_paper         : Paper QuasiMoTTo pure autoregressive arithmetic decoding (no drafter)
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.machinery
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
import dspark_fast
from dspark_paper import DSparkDrafter, ND
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path('/home/dev/KAP/passk/sample')
OUT.mkdir(parents=True, exist_ok=True)
PAR = '/home/dev/KAP/parameter'

B = 64
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
T = 1.0
SEED = 42

SETTINGS = {
    'S1_iid':           dict(kind='classic', draft='iid',     accept='iid',     bonus='iid'),
    'S2_draft':         dict(kind='classic', draft='lattice', accept='iid',     bonus='iid'),
    'S3_draft_acc':     dict(kind='classic', draft='lattice', accept='lattice', bonus='iid'),
    'S4_draft_acc_bns': dict(kind='classic', draft='lattice', accept='lattice', bonus='lattice'),
    'S5_dspark_ac':     dict(kind='dspark_ac'),
    'S6_paper':         dict(kind='paper'),
}

# ---------------- legacy (classic acceptance) engine ----------------
_loader = importlib.machinery.SourceFileLoader(
    'dspark_fast_legacy', '/home/dev/KAP/src4dspark/dspark_fast.py.bak_spec')
_spec = importlib.util.spec_from_loader('dspark_fast_legacy', _loader)
legacy = importlib.util.module_from_spec(_spec)
sys.modules['dspark_fast_legacy'] = legacy
_loader.exec_module(legacy)


def as_tok(x, dev):
    return x if torch.is_tensor(x) else torch.tensor(int(x), device=dev)


def lattice(Bn, gen, dev):
    d = torch.rand((), device=dev, generator=gen)
    return (d + torch.arange(Bn, device=dev, dtype=torch.float32) / Bn) % 1.0


class SettingEngine(legacy.DSparkDecodeFast):
    """Classic min(1, p/q) speculative engine with configurable (u, v, w) stream coupling."""

    def __init__(self, *a, cfg, **kw):
        super().__init__(*a, **kw)
        self.cfg = cfg

    @torch.inference_mode()
    def propose_multi_u(self, feats1, anchor, anchor_pos, n, Bn, u=None):
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        if not torch.is_tensor(anchor):
            anchor = torch.tensor(anchor, device=self.dev)
        if anchor.numel() == 1:
            anc = anchor.reshape(1, 1).expand(1, Bn)
        else:
            anc = anchor.view(1, Bn)
        o = torch.full((1, Bn), anchor_pos - 1, dtype=torch.long, device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anc), o)[0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()
        if u is None:
            u = torch.rand(Bn, device=self.dev, generator=self.gen)
        prev = anc.view(Bn)
        Lg = torch.empty(Bn, n, V, device=self.dev)
        ids = torch.empty(Bn, n, dtype=torch.long, device=self.dev)
        lps = torch.zeros(Bn, device=self.dev)
        for j in range(n):
            lgj = (base[:, j] + self.d.markov(prev)) / self.T
            Lg[:, j] = lgj
            q = torch.softmax(lgj, -1)
            cdf = q.cumsum(-1)
            xj = torch.searchsorted(cdf, u.unsqueeze(1), right=True).squeeze(1).clamp(max=V - 1)
            qx = q.gather(1, xj.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            lo = cdf.gather(1, xj.unsqueeze(1)).squeeze(1) - qx
            lps += torch.log(qx)
            u = ((u - lo) / qx).clamp(0.0, 1.0 - 1e-7)
            ids[:, j] = xj
            prev = xj
        return Lg, ids, lps, u

    def accept_beam(self, p_lg, q_lg, drafts, v):
        n = drafts.shape[0]
        p = torch.softmax(p_lg / self.T, -1)
        q = torch.softmax(q_lg, -1)
        for t in range(n):
            d = int(drafts[t])
            a = min(1.0, float(p[t, d]) / max(float(q[t, d]), 1e-12))
            vt = v if v is not None else float(torch.rand((), device=self.dev, generator=self.gen))
            if vt < a:
                if v is not None:
                    v = min(max(vt / a, 0.0), 1.0 - 1e-7) if a > 0 else 0.0
                continue
            wv = (vt - a) / max(1.0 - a, 1e-12)
            res = (p[t] - q[t]).clamp_min(0)
            Z = float(res.sum())
            if Z <= 0.0:
                return t, int(p[t].argmax()), v
            cdf = res.cumsum(-1) / Z
            c = int(torch.searchsorted(cdf, torch.tensor(wv, device=self.dev),
                                       right=True).clamp(max=p.shape[-1] - 1))
            rc = float(res[c]) / Z
            lo = float(cdf[c]) - rc
            if v is not None:
                v = min(max((wv - lo) / max(rc, 1e-12), 0.0), 1.0 - 1e-7)
            return t, c, v
        return n, None, v

    def sample_w(self, row, w):
        p = torch.softmax(row.float() / self.T, -1)
        cdf = p.cumsum(-1)
        x = int(torch.searchsorted(cdf, torch.tensor(float(w), device=self.dev),
                                   right=True).clamp(max=p.numel() - 1))
        qx = float(p[x].clamp_min(1e-12))
        lo = float(cdf[x]) - qx
        w = min(max((float(w) - lo) / qx, 0.0), 1.0 - 1e-7)
        return x, w

    @torch.inference_mode()
    def generate_settings(self, prompt_ids, max_new_tokens, n=ND, branches=B):
        cfg = self.cfg
        Bn = branches
        ids_t = torch.tensor([prompt_ids], device=self.dev)
        cache0, last0, feats0 = self.prefill(ids_t)
        pos0 = ids_t.shape[1]

        # Round-1 anchors: draw INDEPENDENTLY per branch when bonus=='iid'
        # (previously drew a single anchor and cloned it across all 64 branches!)
        if cfg['bonus'] == 'lattice':
            w = lattice(Bn, self.gen, self.dev)
            p0 = torch.softmax(last0.float() / self.T, -1)
            cdf0 = p0.cumsum(-1)
            anchors = torch.searchsorted(cdf0, w.unsqueeze(1), right=True).squeeze(1).clamp(max=p0.numel() - 1)
            qx = p0[anchors].clamp_min(1e-12)
            lo = cdf0[anchors] - qx
            w = ((w - lo) / qx).clamp(0.0, 1.0 - 1e-7).tolist()
        else:
            p0 = torch.softmax(last0.float() / self.T, -1)
            cdf0 = p0.cumsum(-1)
            u_anc = torch.rand(Bn, device=self.dev, generator=self.gen)
            anchors = torch.searchsorted(cdf0, u_anc.unsqueeze(1), right=True).squeeze(1).clamp(max=p0.numel() - 1)
            w = None

        u_d = lattice(Bn, self.gen, self.dev) if cfg['draft'] == 'lattice' else None
        Lg, dr, lps, u_res = self.propose_multi_u(feats0[0], anchors, pos0, n, Bn, u=u_d)

        snap = self.rw.snap(cache0)
        self._expand_cache(cache0, Bn)
        chunk = torch.cat([anchors.view(Bn, 1), dr], 1)
        v_lg, v_feats, hs, cache_exp = self._forward(cache0, chunk, pos0)

        v_init = lattice(Bn, self.gen, self.dev) if cfg['accept'] == 'lattice' else None
        ks, corrs, vstates = [], [], []
        for b in range(Bn):
            vb = float(v_init[b]) if v_init is not None else None
            k, corr, vnew = self.accept_beam(v_lg[b, :n], Lg[b], dr[b], vb)
            ks.append(k)
            corrs.append(corr)
            vstates.append(vnew)

        beams = []
        for b in range(Bn):
            k = ks[b]
            if k == n:
                c = self._branch_cache(cache_exp, b)
                feats = torch.cat([feats0, v_feats[b:b + 1]], 1)
                pos = pos0 + n + 1
                committed = chunk[b]
                if cfg['bonus'] == 'lattice':
                    anc, w[b] = self.sample_w(v_lg[b, n], w[b])
                else:
                    anc = self._sample(v_lg[b, n])
                anc = as_tok(anc, self.dev)
            else:
                c = self._branch_cache(cache_exp, b)
                self.rw.restore(c, snap, n_crop=(n + 1) - (k + 1))
                self.rw.rewind(c, snap, hs, k + 1, bidx=b)
                feats = torch.cat([feats0, v_feats[b:b + 1, :k + 1]], 1)
                pos = pos0 + k + 1
                committed = chunk[b, :k + 1]
                anc = as_tok(corrs[b], self.dev)
            beams.append(dict(cache=c, feats=feats, anchor=anc, pos=pos,
                              u=(u_res[b] if cfg['draft'] == 'lattice' else None),
                              v=vstates[b], w=(w[b] if w is not None else None),
                              out=[committed], acc=[k]))
        del cache_exp
        torch.cuda.empty_cache()

        for bm in beams:
            cache, feats, anchor, pos = bm['cache'], bm['feats'], bm['anchor'], bm['pos']
            u_b, out, acc = bm['u'], bm['out'], bm['acc']
            if bool((out[0] == self.tok.eos_token_id).any()):
                bm['final'] = out[0][:max_new_tokens].tolist()
                bm['accepted'] = acc
                continue
            while sum(t.numel() for t in out) < max_new_tokens:
                if bool((anchor == self.tok.eos_token_id).any()):
                    out.append(anchor.view(1))
                    break
                u_in = u_b if u_b is not None else torch.rand((), device=self.dev, generator=self.gen)
                Lg1, dr1, _, u_out = self._propose_single(feats[0], anchor, pos, n, u_in)
                if u_b is not None:
                    u_b = u_out
                snap1 = self.rw.snap(cache)
                chunk1 = torch.cat([anchor.view(1), dr1]).view(1, -1)
                v_lg1, v_feats1, hs1, cache = self._forward(cache, chunk1, pos)
                k1, corr1, bm['v'] = self.accept_beam(v_lg1[0, :n], Lg1, dr1, bm['v'])
                acc.append(k1)
                if corr1 is None:
                    feats = torch.cat([feats, v_feats1[0:1]], 1)
                    pos += n + 1
                    committed1 = chunk1[0]
                    if cfg['bonus'] == 'lattice':
                        anc1, bm['w'] = self.sample_w(v_lg1[0, n], bm['w'])
                        anchor_next = as_tok(anc1, self.dev)
                    else:
                        anchor_next = self._sample(v_lg1[0, n])
                else:
                    committed1 = chunk1[0, :k1 + 1]
                    self.rw.restore(cache, snap1, n_crop=(n + 1) - (k1 + 1))
                    self.rw.rewind(cache, snap1, hs1, k1 + 1, bidx=0)
                    feats = torch.cat([feats, v_feats1[0:1, :k1 + 1]], 1)
                    pos += k1 + 1
                    anchor_next = as_tok(corr1, self.dev)
                anchor = anchor_next
                out.append(committed1)
                if bool((committed1 == self.tok.eos_token_id).any()):
                    break
                if bool((anchor == self.tok.eos_token_id).any()):
                    out.append(anchor.view(1))
                    break
            else:
                if sum(t.numel() for t in out) < max_new_tokens:
                    out.append(anchor.view(1))
            bm['final'] = torch.cat(out)[:max_new_tokens].tolist()
            bm['accepted'] = acc
        return [(bm['final'], {'accepted': bm['accepted']}) for bm in beams]


# ---------------- paper-style batched pure arithmetic decode (S6) ----------------
@torch.inference_mode()
def batched_pure_decode(model, tok, prompt_ids, u0, max_new, T=T, stop_on_bracket=False):
    dev = model.device
    Bn = u0.shape[0]
    ids = torch.tensor([prompt_ids], device=dev).expand(Bn, -1).contiguous()
    o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    cache, lg = o.past_key_values, o.logits[:, -1].float()
    pos = ids.shape[1]
    u = u0.clone().float()
    outs = [[] for _ in range(Bn)]
    done = torch.zeros(Bn, dtype=torch.bool, device=dev)
    rows = torch.arange(Bn, device=dev)
    for _ in range(max_new):
        p = torch.softmax(lg / T, -1)
        cdf = p.cumsum(-1)
        x = torch.searchsorted(cdf, u.unsqueeze(1), right=True).squeeze(1).clamp(max=p.shape[-1] - 1)
        qx = p[rows, x].clamp_min(1e-12)
        lo = cdf[rows, x] - qx
        u = ((u - lo) / qx).clamp(0.0, 1.0 - 1e-7)
        for b in range(Bn):
            if not done[b]:
                tid = int(x[b])
                outs[b].append(tid)
                if stop_on_bracket and ']' in tok.decode([tid]):
                    done[b] = True
        done = done | (x == tok.eos_token_id)
        if done.all():
            break
        o = model(input_ids=x.unsqueeze(1), past_key_values=cache,
                  position_ids=torch.full((Bn, 1), pos, dtype=torch.long, device=dev),
                  use_cache=True, logits_to_keep=1)
        cache, lg = o.past_key_values, o.logits[:, -1].float()
        pos += 1
    return outs


@torch.inference_mode()
def answer_prob(model, tok, prompt_ids, answer_text, T=T, include_eos=True):
    """Exact teacher-forced sequence probability pi_theta(answer | prompt)."""
    dev = model.device
    ans = tok(answer_text, add_special_tokens=False).input_ids
    if include_eos and tok.eos_token_id is not None:
        ans = ans + [tok.eos_token_id]
    ids = torch.tensor([prompt_ids + ans], device=dev)
    o = model(input_ids=ids)
    L = len(ans)
    lg = o.logits[0, -(L + 1):-1].float() / T
    lps = torch.log_softmax(lg, -1)
    ans_t = torch.tensor(ans, device=dev)
    lp = float(lps.gather(1, ans_t.unsqueeze(1)).sum())
    return math.exp(lp)


# ---------------- unbiased estimators (QuasiMoTTo Sec. 2.4.1 & Thm 1) ----------------
def std_est(N, c, k):
    """Unbiased estimator of 1 - (1 - p)^k for i.i.d. rollouts (Chen et al. 2021)."""
    return 1.0 if N - c < k else 1.0 - math.comb(N - c, k) / math.comb(N, k)


def lattice_est(hits, k):
    """QuasiMoTTo Theorem 1: unbiased dyadic stride bootstrap for lattice rollouts."""
    N = len(hits)
    stride = N // k
    vals = [1.0 if any(hits[j] for j in range(r, N, stride)) else 0.0 for r in range(stride)]
    return sum(vals) / len(vals)


# ---------------- grading & dataset builders ----------------
def construct_1d_arc_prompt(task: dict) -> str:
    lines = []
    for ex in task['train']:
        lines.append(f"Input: {ex['input'][0]}")
        lines.append(f"Output: {ex['output'][0]}")
        lines.append("")
    lines.append(f"Input: {task['test'][0]['input'][0]}")
    lines.append("Output:")
    return "\n".join(lines)


def check_resp(domain: str, resp: str, it: dict) -> bool:
    if domain == 'sudoku':
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.S | re.I)
        if m:
            resp = m.group(1)
        resp = resp.strip()
        if not re.fullmatch(r'[\s\d]*', resp):
            return False
        nums = re.findall(r'[1-9]', resp)
        if len(nums) != 81:
            return False
        return [int(x) for x in nums] == [int(x) for x in re.findall(r'[1-9]', it['raw']['sol'])]

    if domain == 'maze':
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.S | re.I)
        if m:
            resp = m.group(1)
        path = resp.strip()
        if not path or any(c not in 'URDL' for c in path):
            return False
        g = [list(r) for r in it['maze_grid'].strip().splitlines()]
        H, W = len(g), len(g[0])
        si = sc = ei = ec = None
        for i in range(H):
            for j in range(W):
                if g[i][j] == 'S':
                    si, sc = i, j
                elif g[i][j] == 'E':
                    ei, ec = i, j
        r, c = si, sc
        dd = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        for ch in path:
            dr, dc = dd[ch]
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W) or g[r][c] == '*':
                return False
        return (r, c) == (ei, ec)

    if domain == 'countdown':
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.S | re.I)
        if m:
            resp = m.group(1)
        nums, target = it['raw']['nums'], it['raw']['target']
        for c_ in re.findall(r'[\d\(\)\+\-\*\/][\d\(\)\+\-\*\/\s]*[\d\)]', resp):
            if '**' in c_ or '//' in c_:
                continue
            if not re.search(r'[\+\-\*\/]', c_):
                continue
            runs = re.findall(r'\d+', c_)
            if sorted(map(int, runs)) != sorted(nums):
                continue
            try:
                v = eval(c_, {'__builtins__': {}}, {})
            except Exception:
                continue
            if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                v = round(v)
            if v == target:
                return True
        return False

    if domain == '1d_arc':
        m = re.search(r'\[.*?\]', resp, re.S)
        if not m:
            return False
        try:
            pred = json.loads(m.group())
            return isinstance(pred, list) and [int(x) for x in pred] == it['truth']
        except Exception:
            return False
    return False


DOMAIN_CFG = {
    'sudoku': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model',
        drafter=PAR + '/draft/dspark_sudoku_sft/best.pt',
        max_new=200,
        ub_mode='gt_no_eos',  # matches QuasiMoTTo eval.py lines 148-149
    ),
    'maze': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-maze-sft/hf_model',
        drafter=PAR + '/draft/dspark_maze_sft/best.pt',
        max_new=64,
        ub_mode='gt_no_eos',  # matches QuasiMoTTo eval.py lines 148-149
    ),
    'countdown': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model',
        drafter=PAR + '/draft/dspark_countdown_sft/best.pt',
        max_new=64,
        ub_mode='mc',
    ),
    # Per QuasiMoTTo Sec. 4 ("Models"): 1D-ARC is inductive in-context and must NOT be
    # SFT-memorized on the 901 task files; evaluate directly on Base + domain drafter.
    '1d_arc': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-original',
        drafter=PAR + '/draft/dspark_domain/best.pt',
        max_new=200,
        ub_mode='gt_no_eos',
    ),
}


def load_domain_prompts(domain: str, tok, n_prompts: int):
    rng = random.Random(SEED + 1)
    if domain == 'sudoku':
        items = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'sudoku' and it['split'] == 'val']
        with open('/home/dev/KAP/data/maze_sudoku/sudoku_val.csv') as f:
            gmap = {r['sol']: r['puzzle'] for r in csv.DictReader(f)}
        val = [dict(it, raw={**it['raw'], 'puzzle': gmap[it['raw']['sol']]})
               for it in val if it['raw']['sol'] in gmap]
        return rng.sample(val, min(n_prompts, len(val)))

    if domain == 'maze':
        items = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'maze' and it['split'] == 'val']
        rows = list(csv.DictReader(open('/home/dev/KAP/data/maze_sudoku/maze_val.csv')))
        val.sort(key=lambda x: x['id'])
        base = val[0]['id']
        val = [dict(it, maze_grid=rows[it['id'] - base]['maze'])
               for it in val if 0 <= it['id'] - base < len(rows)]
        return rng.sample(val, min(n_prompts, len(val)))

    if domain == 'countdown':
        items = torch.load('/home/dev/KAP/data/domain_exp/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'countdown' and it['split'] == 'val']
        # Note: sft_train.py trained countdown with "Target: ...\nNumbers: ...\n"
        out = []
        for it in val:
            raw = it['raw']
            p_str = f"Target: {raw['target']}\nNumbers: {raw['nums']}\n"
            out.append(dict(it, ids=tok(p_str).input_ids))
        return rng.sample(out, min(n_prompts, len(out)))

    if domain == '1d_arc':
        files = sorted(glob.glob('/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset/*/*.json'))
        picked = rng.sample(files, min(n_prompts, len(files)))
        out = []
        for f in picked:
            task = json.load(open(f))
            p_str = construct_1d_arc_prompt(task)
            truth = task['test'][0]['output'][0]
            out.append({
                'ids': tok(p_str).input_ids,
                'truth': truth,
                'ans_str': ' ' + json.dumps(truth),  # matches eval.py line 245
                'src': f,
            })
        return out
    raise ValueError(domain)


def plot_results(domain: str, results: dict, fig_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    COLORS = {
        'S1_iid':           '#eb6834',
        'S2_draft':         '#eda100',
        'S3_draft_acc':     '#8c6bb1',
        'S4_draft_acc_bns': '#e87ba4',
        'S5_dspark_ac':     '#1baf7a',
        'S6_paper':         '#2a78d6',
    }
    STYLES = {
        'S1_iid':           ('s', '--'),
        'S2_draft':         ('^', '--'),
        'S3_draft_acc':     ('v', ':'),
        'S4_draft_acc_bns': ('D', '--'),
        'S5_dspark_ac':     ('o', '-'),
        'S6_paper':         ('*', '-'),
    }
    LABELS = {
        'S1_iid':           'S1: classic all i.i.d. (std est.)',
        'S2_draft':         'S2: classic coupled draft u',
        'S3_draft_acc':     'S3: classic coupled u + accept v',
        'S4_draft_acc_bns': 'S4: classic coupled u + v + bonus w',
        'S5_dspark_ac':     'S5: DSpark arithmetic-coding (exact Φ_p(u))',
        'S6_paper':         'S6: QuasiMoTTo paper (no drafter)',
    }

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    for name in SETTINGS:
        if name not in results:
            continue
        curve = results[name]['curve']
        ks = sorted(int(k) for k in curve)
        ys = [curve[str(k)] if str(k) in curve else curve[k] for k in ks]
        mk, ls = STYLES[name]
        tau_info = results[name].get('mean_tau')
        lbl = LABELS[name] + (f'  [τ={tau_info:.2f}]' if tau_info else '')
        lw = 2.4 if name in ('S5_dspark_ac', 'S6_paper') else 1.7
        ax.plot(ks, [y * 100 for y in ys], color=COLORS[name], ls=ls, lw=lw,
                marker=mk, markersize=6, markeredgewidth=0, label=lbl,
                zorder=4 if name == 'S5_dspark_ac' else 3)

    if 'bound' in results:
        b = results['bound']
        ks_b = sorted(int(k) for k in b)
        ys_b = [b[str(k)] if str(k) in b else b[k] for k in ks_b]
        ax.plot(ks_b, [y * 100 for y in ys_b], color='#1e1e1c', ls=':', lw=2.0,
                label='Upper bound E[min(kp, 1)]', zorder=2)

    if 'iid_theory' in results:
        it_c = results['iid_theory']
        ks_i = sorted(int(k) for k in it_c)
        ys_i = [it_c[str(k)] if str(k) in it_c else it_c[k] for k in ks_i]
        ax.plot(ks_i, [y * 100 for y in ys_i], color='#898781', ls='-.', lw=1.5,
                label='Theory i.i.d. E[1 − (1 − p)^k]', zorder=2)

    n_p = results['config']['n_prompts']
    ax.set_title(f'{domain.upper()} pass@k across sampling settings  ({n_p} prompts × {B} rollouts, n={ND})',
                 fontsize=10.5, color='#0b0b0b', fontweight='semibold', loc='left', pad=8)
    ax.set_xlabel('k (samples)', fontsize=9.5, color='#555550')
    ax.set_ylabel('pass@k (%)', fontsize=9.5, color='#555550')
    ax.set_xscale('log', base=2)
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', color='#e1e0d9', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#c0bfb8')
    ax.tick_params(colors='#555550', labelsize=8.5)
    ax.legend(fontsize=7.8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    print('saved', fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='sudoku', choices=['sudoku', 'maze', 'countdown', '1d_arc'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-prompts', type=int, default=100)
    ap.add_argument('--settings', nargs='+', default=list(SETTINGS.keys()))
    ap.add_argument('--merge-into', type=str, default=None,
                    help='optional path to existing JSON to merge computed settings into')
    ap.add_argument('--out-json', type=str, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()

    dom = args.domain
    dev = args.device
    dcfg = DOMAIN_CFG[dom]
    max_new = dcfg['max_new']
    n_prompts = 2 if args.smoke else args.n_prompts

    suffix = '' if dom == 'sudoku' else f'_{dom}'
    if args.smoke:
        suffix += '_smoke'
    res_path = Path(args.out_json) if args.out_json else (OUT / f'results{suffix}.json')
    fig_path = OUT / f'passk_settings{suffix}.png'

    if args.plot_only:
        results = json.load(open(res_path))
        if args.merge_into:
            other = json.load(open(args.merge_into))
            for k, v in other.items():
                if k != 'config':
                    results[k] = v
            json.dump(results, open(res_path, 'w'), indent=2)
        plot_results(dom, results, fig_path)
        return

    tok = AutoTokenizer.from_pretrained(dcfg['target'])
    model = AutoModelForCausalLM.from_pretrained(
        dcfg['target'], dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    drafter = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128,
                            device=dev, dtype=torch.float32)
    drafter.load_state_dict(
        torch.load(dcfg['drafter'], map_location=dev, weights_only=False)['model'])
    drafter.eval()

    prompts = load_domain_prompts(dom, tok, n_prompts)
    n_prompts = len(prompts)
    print(f'=== {dom.upper()}: {n_prompts} prompts, B={B}, max_new={max_new}, device={dev} ===', flush=True)

    results = {
        'config': dict(
            domain=dom, B=B, n=ND, max_new=max_new, n_prompts=n_prompts,
            temperature=T,
            estimator='Chen et al. unbiased 1-C(N-c,k)/C(N,k) for S1_iid; QuasiMoTTo Thm-1 dyadic lattice bootstrap for S2-S6',
        )
    }

    t_all = time.time()
    stop_bracket = (dom == '1d_arc')

    for name in args.settings:
        scfg = SETTINGS[name]
        t0 = time.time()
        hits_all = []
        taus_all = []
        curves = {k: [] for k in K_VALUES}
        curves_std = {k: [] for k in K_VALUES}

        for i, it in enumerate(prompts):
            gen = torch.Generator(device=dev).manual_seed(SEED * 1000 + i)
            if scfg['kind'] == 'paper':
                u0 = lattice(B, gen, dev)
                seqs = batched_pure_decode(model, tok, it['ids'], u0, max_new,
                                           T=T, stop_on_bracket=stop_bracket)
                hits = [int(check_resp(dom, tok.decode(s, skip_special_tokens=True), it))
                        for s in seqs]
            elif scfg['kind'] == 'dspark_ac':
                u0 = lattice(B, gen, dev)
                eng = dspark_fast.DSparkDecodeFast(model, drafter, tok, temperature=T, seed=SEED * 1000 + i)
                chains = eng.generate_branches(it['ids'], max_new, n=ND, branches=B,
                                               mode='lattice', return_all=True, u0=u0)
                hits = [int(check_resp(dom, tok.decode(x, skip_special_tokens=True), it))
                        for x, _ in chains]
                for _, st in chains:
                    if st['accepted']:
                        taus_all.append(sum(st['accepted']) / len(st['accepted']) + 1.0)
                del eng
                torch.cuda.empty_cache()
            else:
                eng = SettingEngine(model, drafter, tok, temperature=T, seed=SEED * 1000 + i, cfg=scfg)
                chains = eng.generate_settings(it['ids'], max_new, n=ND, branches=B)
                hits = [int(check_resp(dom, tok.decode(x, skip_special_tokens=True), it))
                        for x, _ in chains]
                for _, st in chains:
                    if st['accepted']:
                        taus_all.append(sum(st['accepted']) / len(st['accepted']) + 1.0)
                del eng
                torch.cuda.empty_cache()

            hits_all.append(hits)
            c = sum(hits)
            use_std = (name == 'S1_iid')
            for k in K_VALUES:
                s_val = std_est(B, c, k)
                l_val = lattice_est(hits, k)
                curves_std[k].append(s_val)
                curves[k].append(s_val if use_std else l_val)

            if (i + 1) % max(1, n_prompts // 5) == 0 or i == 0:
                print(f'  [{name}] prompt {i+1}/{n_prompts} hits={c}/{B}', flush=True)

        rec = {
            'hits': hits_all,
            'curve': {str(k): round(sum(v) / len(v), 4) for k, v in curves.items()},
            'curve_std': {str(k): round(sum(v) / len(v), 4) for k, v in curves_std.items()},
            'sec': round(time.time() - t0, 1),
        }
        if taus_all:
            rec['mean_tau'] = round(sum(taus_all) / len(taus_all), 3)
        results[name] = rec
        tau_str = f'  tau={rec["mean_tau"]:.2f}' if 'mean_tau' in rec else ''
        print(f'[{name}] ' + '  '.join(f'pass@{k}={rec["curve"][str(k)]*100:.1f}%' for k in K_VALUES)
              + f'{tau_str}  ({rec["sec"]}s)', flush=True)

    # ---------------- Upper bound min(kp, 1) & theoretical i.i.d. 1-(1-p)^k ----------------
    ub_p = []
    if dcfg['ub_mode'] == 'gt_eos':
        for it in prompts:
            ans_s = it['ans_str'] if 'ans_str' in it else it['raw']['sol']
            ub_p.append(answer_prob(model, tok, it['ids'], ans_s, T=T, include_eos=True))
    elif dcfg['ub_mode'] == 'gt_no_eos':
        for it in prompts:
            ans_s = it['ans_str'] if 'ans_str' in it else it['raw']['sol']
            ub_p.append(answer_prob(model, tok, it['ids'], ans_s, T=T, include_eos=False))
    elif dcfg['ub_mode'] == 'mc':
        # Countdown has multiple valid arithmetic expressions; estimate p_i = Pr(solve) via
        # a high-precision lattice QMC draw of M=128 samples (exact marginal expectation!).
        M_UB = 128
        for i, it in enumerate(prompts):
            gen_ub = torch.Generator(device=dev).manual_seed(999_000 + i)
            u_ub = lattice(M_UB, gen_ub, dev)
            seqs_ub = batched_pure_decode(model, tok, it['ids'], u_ub, max_new, T=T)
            c_ub = sum(check_resp(dom, tok.decode(s, skip_special_tokens=True), it) for s in seqs_ub)
            ub_p.append(c_ub / M_UB)

    results['upper_bound_p'] = [round(v, 6) for v in ub_p]
    results['bound'] = {
        str(k): round(sum(min(1.0, k * p_) for p_ in ub_p) / len(ub_p), 4)
        for k in K_VALUES
    }
    results['iid_theory'] = {
        str(k): round(sum(1.0 - (1.0 - p_) ** k for p_ in ub_p) / len(ub_p), 4)
        for k in K_VALUES
    }
    print('[bound]      ' + '  '.join(f'pass@{k}={results["bound"][str(k)]*100:.1f}%' for k in K_VALUES), flush=True)
    print('[iid_theory] ' + '  '.join(f'pass@{k}={results["iid_theory"][str(k)]*100:.1f}%' for k in K_VALUES), flush=True)

    results['total_sec'] = round(time.time() - t_all, 1)
    json.dump(results, open(res_path, 'w'), indent=2)
    plot_results(dom, results, fig_path)


if __name__ == '__main__':
    main()
