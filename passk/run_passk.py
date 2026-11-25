"""pass@k evaluation for Maze / Sudoku / Countdown / 1D-ARC (2026-11-24 Readme).

Implements exactly the 5 curves of Readme.md section 2:

  S2_update   : DSpark arithmetic-coding deterministic decode (AC), lattice-coupled u.
  S3_paper    : paper QuasiMoTTo pure autoregressive arithmetic decode (no drafter), lattice u.
  S1_iid      : traditional i.i.d. speculative sampling (same AC engine, i.i.d. u0 —
                sequences are exact i.i.d. draws from pi_theta).
  Upper bound : E_i[min(k * p_i, 1)]                         (QuasiMoTTo Theorem 2)
  Theory i.i.d.: E_i[1 - (1 - p_i)^k]

Estimators (Readme.md section 3, QuasiMoTTo arXiv:2607.01179 Sec 2.4.1 / Thm 1-2):
  - S1_iid hits  -> Chen et al. (2021) unbiased combinatorial 1 - C(B-c, k)/C(B, k).
  - lattice hits -> Theorem-1 dyadic stride bootstrap: mean over stride-r subsequences
    hits[r::s], s = B/k  (NOT the i.i.d. formula — that is biased for coupled samples).
  - S2_update / S3_paper use lattice u0 = (Delta + b/B) mod 1, so they use lattice_est.

Per-task p_i (Readme.md section 1):
  - maze / sudoku / 1d_arc: exact teacher-forced pi_theta(answer | prompt), no EOS
    (matches Ref/quasimotto/eval.py lines 147-149 gt_ub path).
  - countdown: high-precision QMC Monte Carlo with M=128 rollouts (multiple valid
    expressions exist per problem, so no single teacher-forced answer).

Usage (one task at a time — improve task by task):
  python run_passk.py --task maze  --device cuda:0            # full run (100 prompts)
  python run_passk.py --task maze --smoke                     # 2-prompt smoke test
  python run_passk.py --task maze --settings S2_update S3_paper   # partial (re)compute
  python run_passk.py --task maze --merge-into results/results_maze.json --settings S1_iid
  python run_passk.py --task maze --plot-only                 # re-plot from JSON
  python run_passk.py --task maze --weak 10                   # 10 weakest prompts
  python run_passk.py --report                                 # cross-task summary table
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
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

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
PAR = '/home/dev/KAP/parameter'

# ---------------- global sampling configuration (Readme section 1) ----------------
B = 64                                   # rollouts per prompt
K_VALUES = [1, 2, 4, 8, 16, 32, 64]      # dyadic k <= B
T = 1.0                                  # temperature
SEED = 42
M_UB_MC = 128                            # countdown QMC rollouts for p_i

SETTINGS = ('S1_iid', 'S2_update', 'S3_paper')

# Readme section 1 table. drafter for 1d_arc is dspark_arc_sft (NOT dspark_domain).
TASK_CFG = {
    'maze': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-maze-sft/hf_model',
        drafter=PAR + '/draft/dspark_maze_sft/best.pt',
        max_new=64,
        ub_mode='gt',        # teacher-forced exact sequence probability
    ),
    'sudoku': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model',
        drafter=PAR + '/draft/dspark_sudoku_sft/best.pt',
        max_new=200,
        ub_mode='gt',
    ),
    'countdown': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model',
        drafter=PAR + '/draft/dspark_countdown_sft/best.pt',
        max_new=64,
        ub_mode='mc',        # QMC Monte Carlo, M=128 (multiple valid expressions)
    ),
    '1d_arc': dict(
        target=PAR + '/target/Qwen3.5-0.8B-Base-original',
        drafter=PAR + '/draft/dspark_arc_sft/best.pt',
        max_new=200,
        ub_mode='gt',
    ),
}


# ---------------- unbiased estimators (Readme section 3) ----------------
def std_est(N, c, k):
    """Chen et al. (2021) unbiased estimator of 1-(1-p)^k for i.i.d. rollouts.

    Unbiased only under independence/exchangeability — Readme 3.1: applying it to
    coupled (lattice) samples is mathematically wrong (systematically underestimates).
    """
    return 1.0 if N - c < k else 1.0 - math.comb(N - c, k) / math.comb(N, k)


def lattice_est(hits, k):
    """QuasiMoTTo Theorem 1 dyadic stride bootstrap for lattice-coupled rollouts.

    The B-length lattice splits exactly into s = B/k stride-s subsequences hits[r::s],
    each distributionally a fresh size-k randomized lattice with uniform shift.
    """
    stride = len(hits) // k
    return sum(1.0 if any(hits[r::stride]) else 0.0 for r in range(stride)) / stride


# ---------------- samplers ----------------
def lattice_u(Bn, gen, dev):
    """u_b = (Delta + b/B) mod 1, Delta ~ Unif[0,1)  (Readme 3.2)."""
    d = torch.rand((), device=dev, generator=gen)
    return (d + torch.arange(Bn, device=dev, dtype=torch.float32) / Bn) % 1.0


def iid_u(Bn, gen, dev):
    return torch.rand(Bn, device=dev, generator=gen)


@torch.inference_mode()
def batched_pure_decode(model, tok, prompt_ids, u0, max_new, T=T, stop_on_bracket=False):
    """S3_paper: pure autoregressive arithmetic decode Phi_p(u_b), batched over B.

    Each branch owns one sequence uniform u; every token is F_p^{-1}(u) and u updates
    as the inverse-CDF residual u <- (u - F_p(x))/p(x).
    """
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
def answer_prob(model, tok, prompt_ids, answer_text, T=T):
    """Exact teacher-forced sequence probability pi_theta(answer | prompt), no EOS.

    Matches Ref/quasimotto/eval.py gt_ub (lines 147-149): the answer tokens are
    appended to the prompt and their prompt-logprobs multiplied. No EOS token.
    """
    dev = model.device
    ans = tok(answer_text, add_special_tokens=False).input_ids
    ids = torch.tensor([prompt_ids + ans], device=dev)
    o = model(input_ids=ids)
    L = len(ans)
    lg = o.logits[0, -(L + 1):-1].float() / T
    lps = torch.log_softmax(lg, -1)
    ans_t = torch.tensor(ans, device=dev)
    lp = float(lps.gather(1, ans_t.unsqueeze(1)).sum())
    return math.exp(lp)


# ---------------- prompt loading & grading ----------------
def construct_1d_arc_prompt(task: dict) -> str:
    """Same format as Ref/quasimotto/tasks.py construct_1d_arc_prompt."""
    lines = []
    for ex in task['train']:
        lines.append(f"Input: {ex['input'][0]}")
        lines.append(f"Output: {ex['output'][0]}")
        lines.append("")
    lines.append(f"Input: {task['test'][0]['input'][0]}")
    lines.append("Output:")
    return "\n".join(lines)


def check_resp(task: str, resp: str, it: dict) -> bool:
    """Grade one decoded response against the ground truth item."""
    if task == 'sudoku':
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

    if task == 'maze':
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

    if task == 'countdown':
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

    if task == '1d_arc':
        m = re.search(r'\[.*?\]', resp, re.S)
        if not m:
            return False
        try:
            pred = json.loads(m.group())
            return isinstance(pred, list) and [int(x) for x in pred] == it['truth']
        except Exception:
            return False
    raise ValueError(task)


def load_task_prompts(task: str, tok, n_prompts: int):
    """Validation-split prompts per the Readme section 1 table (never passk_filtered)."""
    rng = random.Random(SEED + 1)
    if task == 'sudoku':
        items = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'sudoku' and it['split'] == 'val']
        with open('/home/dev/KAP/data/maze_sudoku/sudoku_val.csv') as f:
            gmap = {r['sol']: r['puzzle'] for r in csv.DictReader(f)}
        val = [dict(it, raw={**it['raw'], 'puzzle': gmap[it['raw']['sol']]})
               for it in val if it['raw']['sol'] in gmap]
        return rng.sample(val, min(n_prompts, len(val)))

    if task == 'maze':
        items = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'maze' and it['split'] == 'val']
        rows = list(csv.DictReader(open('/home/dev/KAP/data/maze_sudoku/maze_val.csv')))
        val.sort(key=lambda x: x['id'])
        base = val[0]['id']
        val = [dict(it, maze_grid=rows[it['id'] - base]['maze'])
               for it in val if 0 <= it['id'] - base < len(rows)]
        return rng.sample(val, min(n_prompts, len(val)))

    if task == 'countdown':
        items = torch.load('/home/dev/KAP/data/domain_exp/prompts.pt', map_location='cpu', weights_only=False)
        val = [it for it in items if it.get('dataset') == 'countdown' and it['split'] == 'val']
        # sft_train.py trained countdown with "Target: ...\nNumbers: ...\n"
        out = []
        for it in val:
            raw = it['raw']
            p_str = f"Target: {raw['target']}\nNumbers: {raw['nums']}\n"
            out.append(dict(it, ids=tok(p_str).input_ids))
        return rng.sample(out, min(n_prompts, len(out)))

    if task == '1d_arc':
        files = sorted(glob.glob('/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset/*/*.json'))
        picked = rng.sample(files, min(n_prompts, len(files)))
        out = []
        for f in picked:
            task_js = json.load(open(f))
            p_str = construct_1d_arc_prompt(task_js)
            truth = task_js['test'][0]['output'][0]
            out.append({
                'ids': tok(p_str).input_ids,
                'truth': truth,
                'ans_str': ' ' + json.dumps(truth),  # matches eval.py line 245
                'src': f,
            })
        return out
    raise ValueError(task)


def gt_answer_str(task: str, it: dict) -> str:
    """Ground-truth answer text teacher-forced for p_i (ub_mode='gt' tasks)."""
    if 'ans_str' in it:
        return it['ans_str']
    return it['raw']['sol']


# ---------------- one sampling setting over all prompts ----------------
def run_setting(name, model, drafter, tok, task, prompts, max_new, dev):
    """Generate B rollouts per prompt under setting `name`; return per-prompt hits.

    S1_iid   : AC engine with i.i.d. u0      -> hits i.i.d. Bernoulli(p_i)  -> std_est
    S2_update: AC engine with lattice u0     -> coupled                     -> lattice_est
    S3_paper : pure arithmetic decode, lattice u0 (no drafter)              -> lattice_est
    """
    stop_bracket = (task == '1d_arc')
    hits_all, taus_all = [], []
    t0 = time.time()
    for i, it in enumerate(prompts):
        gen = torch.Generator(device=dev).manual_seed(SEED * 1000 + i)
        if name == 'S3_paper':
            u0 = lattice_u(B, gen, dev)
            seqs = batched_pure_decode(model, tok, it['ids'], u0, max_new,
                                       T=T, stop_on_bracket=stop_bracket)
            hits = [int(check_resp(task, tok.decode(s, skip_special_tokens=True), it))
                    for s in seqs]
        else:  # S1_iid / S2_update — identical engine, only the u0 coupling differs
            u0 = iid_u(B, gen, dev) if name == 'S1_iid' else lattice_u(B, gen, dev)
            eng = dspark_fast.DSparkDecodeFast(model, drafter, tok, temperature=T,
                                               seed=SEED * 1000 + i)
            chains = eng.generate_branches(it['ids'], max_new, n=ND, branches=B,
                                           mode='lattice', return_all=True, u0=u0)
            hits = [int(check_resp(task, tok.decode(x, skip_special_tokens=True), it))
                    for x, _ in chains]
            for _, st in chains:
                if st['accepted']:
                    taus_all.append(sum(st['accepted']) / len(st['accepted']) + 1.0)
            del eng
            with torch.cuda.device(dev):        # avoid initializing a stray context on cuda:0
                torch.cuda.empty_cache()
        hits_all.append(hits)
        if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 0:
            print(f'  [{name}] prompt {i+1}/{len(prompts)} hits={sum(hits)}/{B}', flush=True)
    rec = {'hits': hits_all, 'sec': round(time.time() - t0, 1)}
    if taus_all:
        rec['mean_tau'] = round(sum(taus_all) / len(taus_all), 3)
    return rec


def setting_curves(name, hits_all):
    """Per-k averaged curve with the estimator mandated by the coupling (Readme 3)."""
    use_std = (name == 'S1_iid')
    per_prompt = {k: [] for k in K_VALUES}
    for hits in hits_all:
        c = sum(hits)
        for k in K_VALUES:
            per_prompt[k].append(std_est(B, c, k) if use_std else lattice_est(hits, k))
    return {str(k): round(sum(v) / len(v), 4) for k, v in per_prompt.items()}


# ---------------- upper bound & theory i.i.d. (Readme 3.3) ----------------
def compute_p_i(model, tok, task, prompts, max_new, dev):
    """Per-prompt p_i per Readme section 1 (teacher-forced exact, or countdown QMC)."""
    ub_mode = TASK_CFG[task]['ub_mode']
    p = []
    if ub_mode == 'gt':
        for i, it in enumerate(prompts):
            p.append(answer_prob(model, tok, it['ids'], gt_answer_str(task, it), T=T))
            if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 0:
                print(f'  [p_i] prompt {i+1}/{len(prompts)}', flush=True)
    elif ub_mode == 'mc':
        # countdown: multiple valid expressions -> QMC estimate of p_i = Pr(solve)
        for i, it in enumerate(prompts):
            gen_ub = torch.Generator(device=dev).manual_seed(999_000 + i)
            u_ub = lattice_u(M_UB_MC, gen_ub, dev)
            seqs_ub = batched_pure_decode(model, tok, it['ids'], u_ub, max_new, T=T)
            c_ub = sum(check_resp(task, tok.decode(s, skip_special_tokens=True), it) for s in seqs_ub)
            p.append(c_ub / M_UB_MC)
            if (i + 1) % max(1, len(prompts) // 5) == 0 or i == 1:
                print(f'  [p_i] prompt {i+1}/{len(prompts)} p={c_ub}/{M_UB_MC}', flush=True)
    else:
        raise ValueError(ub_mode)
    return p


def bound_curves(p_i):
    """UpperBound(k) = E[min(1, k p_i)];  TheoryIID(k) = E[1-(1-p_i)^k]  (Readme 3.3)."""
    n = len(p_i)
    bound = {str(k): round(sum(min(1.0, k * p) for p in p_i) / n, 4) for k in K_VALUES}
    iid_theory = {str(k): round(sum(1.0 - (1.0 - p) ** k for p in p_i) / n, 4) for k in K_VALUES}
    return bound, iid_theory


# ---------------- plotting (Readme section 2, exactly 5 curves) ----------------
COLORS = {
    'S1_iid':    '#eb6834',
    'S2_update': '#1baf7a',
    'S3_paper':  '#2a78d6',
}
STYLES = {
    'S1_iid':    ('s', '--'),
    'S2_update': ('o', '-'),
    'S3_paper':  ('*', '-'),
}
LABELS = {
    'S1_iid':    'S1: i.i.d. speculative sampling (std est.)',
    'S2_update': 'S2: DSpark arithmetic-coding (update)',
    'S3_paper':  'S3: paper pure AC decode (no drafter)',
}


def _curve_pts(curve):
    ks = sorted(int(k) for k in curve)
    return ks, [curve[str(k)] if str(k) in curve else curve[k] for k in ks]


def plot_results(task, results, fig_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    for name in SETTINGS:
        if name not in results:
            continue
        ks, ys = _curve_pts(results[name]['curve'])
        mk, ls = STYLES[name]
        tau = results[name].get('mean_tau')
        lbl = LABELS[name] + (f'  [τ={tau:.2f}]' if tau else '')
        ax.plot(ks, [y * 100 for y in ys], color=COLORS[name], ls=ls,
                lw=2.4 if name != 'S1_iid' else 1.7, marker=mk, markersize=6,
                markeredgewidth=0, label=lbl, zorder=4 if name == 'S2_update' else 3)

    if 'bound' in results:
        ks, ys = _curve_pts(results['bound'])
        ax.plot(ks, [y * 100 for y in ys], color='#1e1e1c', ls=':', lw=2.0,
                label='Upper bound E[min(kp, 1)]', zorder=2)
    if 'iid_theory' in results:
        ks, ys = _curve_pts(results['iid_theory'])
        ax.plot(ks, [y * 100 for y in ys], color='#898781', ls='-.', lw=1.5,
                label='Theory i.i.d. E[1 − (1 − p)^k]', zorder=2)

    n_p = (results.get('config') or {}).get('n_prompts')
    n_txt = f'{n_p} prompts × ' if n_p else ''
    ax.set_title(f'{task.upper()} pass@k across sampling settings  '
                 f'({n_txt}{B} rollouts, n={ND})',
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


# ---------------- reporting helpers (per-task improvement loop) ----------------
def weak_prompts(task, results, top_n):
    """Print the prompts with the lowest p_i / per-prompt pass@1 — improvement targets."""
    p_i = results.get('upper_bound_p')
    per_prompt = {}
    for name in SETTINGS:
        if name in results and 'hits' in results[name]:
            per_prompt[name] = [sum(h) / len(h) for h in results[name]['hits']]
    n = len(p_i) if p_i else (len(next(iter(per_prompt.values()))) if per_prompt else 0)
    if not n:
        print('no per-prompt data in results')
        return
    order = sorted(range(n), key=lambda i: (p_i[i] if p_i else 1.0))
    hdr = ['idx', 'p_i'] + [f'{s} pass@1' for s in SETTINGS if s in per_prompt]
    print(f'== {task}: {top_n} weakest prompts (of {n}) ==')
    print('  '.join(f'{h:>12}' for h in hdr))
    for i in order[:top_n]:
        row = [str(i), f'{p_i[i]:.4f}' if p_i and i < len(p_i) else '-']
        row += [f'{per_prompt[s][i]:.4f}' if s in per_prompt and i < len(per_prompt[s]) else '-'
                for s in SETTINGS]
        print('  '.join(f'{v:>12}' for v in row))


def report_all():
    """Cross-task summary from every results/results_*.json (track improvements over time)."""
    files = sorted(OUT.glob('results_*.json'))
    if not files:
        print(f'no results_*.json under {OUT}')
        return
    names = ['S1_iid', 'S2_update', 'S3_paper', 'bound', 'iid_theory']

    def val(r, n, k):
        rec = r.get(n)
        if not isinstance(rec, dict):
            return None
        c = rec.get('curve', rec)          # settings store {curve:{...}}; bound/iid_theory are flat
        v = c.get(str(k))
        return None if v is None else v * 100

    hdr = ['task', 'n_prompt'] + [f'{n}@64' for n in names] + [f'{n}@1' for n in names[:3]]
    print('  '.join(f'{h:>14}' for h in hdr))
    for f in files:
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f'{f.name}: unreadable ({e})')
            continue
        if 'config' not in r:
            continue
        task = r['config'].get('task') or f.stem.removeprefix('results_')
        if f.stem.endswith('_smoke'):
            task += '(smoke)'
        row = [task, str(r['config']['n_prompts'])]
        for n in names:
            v = val(r, n, 64)
            row.append('-' if v is None else f'{v:.1f}%')
        for n in names[:3]:
            v = val(r, n, 1)
            row.append('-' if v is None else f'{v:.1f}%')
        print('  '.join(f'{v:>14}' for v in row))


# ---------------- process title (top/htop comm; nvidia-smi stays default) ----------------
def set_proc_title(name: str):
    """prctl(PR_SET_NAME): rename the current process (<=15 chars) so per-task runs
    are distinguishable in top/htop/ps. nvidia-smi is untouched (it shows argv[0]).
    Best-effort — never fatal."""
    try:
        import ctypes
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=list(TASK_CFG), help='task to evaluate (one at a time)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-prompts', type=int, default=100)
    ap.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    ap.add_argument('--merge-into', type=str, default=None,
                    help='existing results JSON to merge newly computed settings into')
    ap.add_argument('--out-json', type=str, default=None)
    ap.add_argument('--smoke', action='store_true', help='2-prompt smoke test')
    ap.add_argument('--plot-only', action='store_true', help='re-plot from existing JSON')
    ap.add_argument('--weak', type=int, default=0, metavar='N',
                    help='print the N weakest prompts of the task (from existing JSON)')
    ap.add_argument('--report', action='store_true', help='cross-task summary table')
    args = ap.parse_args()

    if args.report:
        report_all()
        return

    if not args.task:
        ap.error('--task is required (choose from: ' + ', '.join(TASK_CFG) + ')')

    task = args.task
    if task:
        set_proc_title(f'passk_{task}')
    dev = args.device
    cfg = TASK_CFG[task]
    max_new = cfg['max_new']
    n_prompts = 2 if args.smoke else args.n_prompts

    suffix = f'_{task}'
    if args.smoke:
        suffix += '_smoke'
    res_path = Path(args.out_json) if args.out_json else (OUT / f'results{suffix}.json')
    fig_path = OUT / f'passk{suffix}.png'
    if not res_path.exists():
        smoke = OUT / f'results{suffix}_smoke.json'
        if args.weak and smoke.exists():       # --weak tolerates smoke-only results
            res_path = smoke
            print(f'[note] {res_path.name} not found; using smoke results')
        elif args.weak or args.plot_only:
            sys.exit(f'no results at {res_path} — run the task first (or pass --out-json)')

    if args.weak:
        results = json.load(open(res_path))
        weak_prompts(task, results, args.weak)
        return

    if args.plot_only:
        results = json.load(open(res_path))
        if args.merge_into:
            other = json.load(open(args.merge_into))
            for k, v in other.items():
                if k != 'config':
                    results[k] = v
            json.dump(results, open(res_path, 'w'), indent=2)
        plot_results(task, results, fig_path)
        return

    tok = AutoTokenizer.from_pretrained(cfg['target'])
    model = AutoModelForCausalLM.from_pretrained(
        cfg['target'], dtype=torch.bfloat16, device_map=dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    drafter = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128,
                            device=dev, dtype=torch.float32)
    drafter.load_state_dict(
        torch.load(cfg['drafter'], map_location=dev, weights_only=False)['model'])
    drafter.eval()

    prompts = load_task_prompts(task, tok, n_prompts)
    n_prompts = len(prompts)
    if not prompts:
        sys.exit(f'no prompts found for task {task} (check --n-prompts and the val split)')
    print(f'=== {task.upper()}: {n_prompts} prompts, B={B}, max_new={max_new}, '
          f'device={dev}, T={T} ===', flush=True)

    results = {
        'config': dict(
            task=task, B=B, n=ND, max_new=max_new, n_prompts=n_prompts, temperature=T,
            seed=SEED,
            estimator='S1_iid: Chen et al. unbiased 1-C(B-c,k)/C(B,k); '
                      'S2_update/S3_paper: QuasiMoTTo Thm-1 dyadic stride lattice bootstrap',
        ),
    }

    t_all = time.time()
    for name in args.settings:
        rec = run_setting(name, model, drafter, tok, task, prompts, max_new, dev)
        rec['curve'] = setting_curves(name, rec['hits'])
        results[name] = rec
        tau_str = f'  tau={rec["mean_tau"]:.2f}' if 'mean_tau' in rec else ''
        print(f'[{name}] ' + '  '.join(f'pass@{k}={rec["curve"][str(k)]*100:.1f}%'
                                       for k in K_VALUES) + f'{tau_str}  ({rec["sec"]}s)',
              flush=True)

    # ---- upper bound & theory i.i.d. ----
    ub_p = compute_p_i(model, tok, task, prompts, max_new, dev)
    results['upper_bound_p'] = [round(v, 6) for v in ub_p]
    results['bound'], results['iid_theory'] = bound_curves(ub_p)
    print('[bound]      ' + '  '.join(f'pass@{k}={results["bound"][str(k)]*100:.1f}%'
                                      for k in K_VALUES), flush=True)
    print('[iid_theory] ' + '  '.join(f'pass@{k}={results["iid_theory"][str(k)]*100:.1f}%'
                                      for k in K_VALUES), flush=True)

    if args.merge_into:
        other = json.load(open(args.merge_into))
        for k, v in other.items():
            if k != 'config' and k not in results:
                results[k] = v

    results['total_sec'] = round(time.time() - t_all, 1)
    json.dump(results, open(res_path, 'w'), indent=2)
    print('saved', res_path)
    plot_results(task, results, fig_path)


if __name__ == '__main__':
    main()
