"""Generate 32 completions per training prompt for each SFT domain,
compute pass@1 = correct/32, and save filtered items where 0 < pass@1 < 0.125.

Usage:
    python gen_passk_filter.py [--domains maze sudoku countdown 1d_arc]
                               [--max-per-domain N]  # 0 = all
                               [--batch-prompts B]   # prompts per generate call
                               [--out-dir /home/dev/KAP/data/passk_filtered]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PAR   = '/home/dev/KAP/parameter'
DATA  = '/home/dev/KAP/data'

DOMAIN_CFG = {
    'maze': {
        'target':   PAR + '/target/Qwen3.5-0.8B-Base-maze-sft/hf_model',
        'data_pt':  DATA + '/maze_sudoku/prompts.pt',
        'train_csv': DATA + '/maze_sudoku/maze_train.csv',
        'max_new':  64,
    },
    'sudoku': {
        'target':   PAR + '/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model',
        'data_pt':  DATA + '/maze_sudoku/prompts.pt',
        'train_csv': DATA + '/maze_sudoku/sudoku_train.csv',
        'max_new':  256,
    },
    'countdown': {
        'target':   PAR + '/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model',
        'data_pt':  DATA + '/domain_exp/prompts.pt',
        'train_csv': None,
        'max_new':  128,
    },
    '1d_arc': {
        'target':   PAR + '/target/Qwen3.5-0.8B-Base-original',
        'data_pt':  DATA + '/domain_exp/prompts.pt',
        'train_csv': None,
        'max_new':  320,
    },
}

# ---------------------------------------------------------------------------
# Verify functions (adapted from passk_sft.ipynb for our raw dict format)
# ---------------------------------------------------------------------------

def _parse_ans(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.S | re.I)
    if m: text = m.group(1).strip()
    m = re.search(r'\\boxed\{', text)
    if m:
        i = m.end(); depth = 1
        for j in range(i, len(text)):
            if text[j] == '{': depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    text = text[i:j].strip(); break
    return text.strip()

def check_maze(resp: str, maze_str: str) -> bool:
    resp = _parse_ans(resp)
    grid = [list(r) for r in maze_str.strip().splitlines()]
    H, W = len(grid), len(grid[0])
    si = sc = ei = ec = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S': si, sc = i, j
            if grid[i][j] == 'E': ei, ec = i, j
    if si is None or ei is None:
        return False
    path = resp
    if not path:
        return False
    # reference grader semantics: every character must be a move (no lowercase); the FINAL
    # position must be the exit (no early return when passing over E)
    if any(ch not in 'URDL' for ch in path):
        return False
    r, c = si, sc
    dirs = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    for m in path:
        dr, dc = dirs[m]; r, c = r + dr, c + dc
        if not (0 <= r < H and 0 <= c < W): return False
        if grid[r][c] == '*': return False
    return (r, c) == (ei, ec)


def check_sudoku(resp: str, sol: str) -> bool:
    comp = _parse_ans(resp)
    if not re.fullmatch(r'[\s\d]*', comp):
        return False
    nums = re.findall(r'[1-9]', comp)
    if len(nums) != 81:
        return False
    pred = [int(x) for x in nums]
    expected = [int(x) for x in re.findall(r'[1-9]', sol)]
    return len(expected) == 81 and pred == expected


def check_countdown(resp: str, nums: list, target: int) -> bool:
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.S | re.I)
    text = m.group(1).strip() if m else resp
    cands = re.findall(r'[\d\(\)\+\-\*\/][\d\(\)\+\-\*\/\s]*[\d\)]', text)
    for c in cands:
        if '**' in c or '//' in c:          # countdown allows only + - * /
            continue
        if not re.search(r'[\+\-\*\/]', c):
            continue
        runs = re.findall(r'\d+', c)
        if sorted(map(int, runs)) != sorted(nums):
            continue
        try:
            val = eval(c, {'__builtins__': {}}, {})
        except Exception:
            continue
        if isinstance(val, float) and abs(val - round(val)) < 1e-9:
            val = round(val)
        if val == target:
            return True
    return False


def check_arc(resp: str, expected_output: list) -> bool:
    truth = expected_output[0]
    width = len(truth)
    m = re.search(r'\[.*?\]', resp, re.S)
    if not m:
        return False
    try:
        pred = json.loads(m.group())
        if not isinstance(pred, list):
            return False
        pred = [int(x) for x in pred]
    except Exception:
        return False
    return pred == truth


def verify_item(ds: str, resp: str, raw: dict, maze_grid: str = None,
                sudoku_puzzle: str = None) -> bool:
    if ds == 'maze':
        return check_maze(resp, maze_grid)
    if ds == 'sudoku':
        return check_sudoku(resp, raw['sol'])
    if ds == 'countdown':
        return check_countdown(resp, raw['nums'], raw['target'])
    if ds == '1d_arc':
        return check_arc(resp, raw['test']['output'])
    return False


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_items(ds: str, cfg: dict, max_n: int) -> list:
    """Return list of dicts with 'ids', 'raw', and optionally 'maze_grid' / 'sudoku_puzzle'."""
    all_items = torch.load(cfg['data_pt'], map_location='cpu', weights_only=False)
    items = [it for it in all_items if it['dataset'] == ds and it['split'] == 'train']
    if max_n and max_n < len(items):
        items = items[:max_n]

    if ds == 'maze' and cfg['train_csv']:
        rows = list(csv.DictReader(open(cfg['train_csv'])))
        items.sort(key=lambda x: x['id'])
        base = items[0]['id'] if items else 0
        # prompts.pt row order == CSV row order (verified 20k/20k); the 'sol'
        # column is NOT unique, so index by row, never by sol value
        items = [dict(it, maze_grid=rows[it['id'] - base]['maze'])
                 for it in items if 0 <= it['id'] - base < len(rows)]

    if ds == 'sudoku' and cfg['train_csv']:
        with open(cfg['train_csv']) as f:
            sol2puzzle = {row['sol']: row['puzzle'] for row in csv.DictReader(f)}
        items = [dict(it, sudoku_puzzle=sol2puzzle.get(it['raw']['sol'], ''))
                 for it in items if it['raw']['sol'] in sol2puzzle]

    return items


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_completions(model, tokenizer, prompt_ids: list, max_new: int,
                         n: int = 32, temperature: float = 1.0,
                         device: str = 'cuda:0') -> list[list[str]]:
    """For each prompt (list of token ids), generate n completions.
    Returns list[list[str]] of shape [len(prompt_ids), n]."""
    results = []
    for ids in prompt_ids:
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        input_ids = input_ids.expand(n, -1)
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[:, input_ids.shape[1]:]
        texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]
        results.append(texts)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_domain(ds: str, cfg: dict, out_dir: Path, max_n: int,
               batch_prompts: int, device: str = 'cuda:0'):
    out_path = out_dir / f'{ds}_filtered.pt'
    if out_path.exists():
        print(f'[{ds}] {out_path} already exists — skipping. Delete to re-run.')
        return

    print(f'\n{"="*60}\n  {ds.upper()}\n{"="*60}')
    items = load_items(ds, cfg, max_n)
    print(f'  {len(items)} train items loaded')

    model = AutoModelForCausalLM.from_pretrained(
        cfg['target'], dtype=torch.bfloat16, device_map=device).eval()
    tok = AutoTokenizer.from_pretrained(cfg['target'])

    filtered = []
    t0 = time.perf_counter()
    n_done = 0

    for b_start in range(0, len(items), batch_prompts):
        batch = items[b_start: b_start + batch_prompts]
        prompt_ids_list = [it['ids'] for it in batch]

        completions = generate_completions(
            model, tok, prompt_ids_list,
            max_new=cfg['max_new'], n=32, temperature=1.0, device=device)

        for it, comps in zip(batch, completions):
            correct = sum(
                verify_item(
                    ds, resp, it['raw'],
                    maze_grid=it.get('maze_grid', ''),
                    sudoku_puzzle=it.get('sudoku_puzzle', ''))
                for resp in comps)
            p1 = correct / 32
            if 0 < p1 < 0.125:
                filtered.append({**it, 'pass1': p1, 'n_correct': correct,
                                  'completions': comps})

        n_done += len(batch)
        elapsed = time.perf_counter() - t0
        rate = n_done / elapsed
        eta = (len(items) - n_done) / rate if rate > 0 else 0
        print(f'  [{ds}] {n_done}/{len(items)}  filtered={len(filtered)}'
              f'  {rate:.1f} it/s  ETA {eta/60:.0f}m', flush=True)

    torch.save(filtered, out_path)
    print(f'  [{ds}] saved {len(filtered)} items → {out_path}')

    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+',
                    default=['maze', 'sudoku', 'countdown', '1d_arc'])
    ap.add_argument('--max-per-domain', type=int, default=0,
                    help='0 = all train items')
    ap.add_argument('--batch-prompts', type=int, default=4,
                    help='prompts per generate call (each × 32 completions)')
    ap.add_argument('--out-dir', default='/home/dev/KAP/data/passk_filtered')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.domains:
        if ds not in DOMAIN_CFG:
            print(f'Unknown domain {ds!r}, skipping')
            continue
        run_domain(ds, DOMAIN_CFG[ds], out_dir,
                   max_n=args.max_per_domain,
                   batch_prompts=args.batch_prompts,
                   device=args.device)

    # Summary
    print('\n--- Summary ---')
    for ds in args.domains:
        p = out_dir / f'{ds}_filtered.pt'
        if p.exists():
            items = torch.load(p, map_location='cpu', weights_only=False)
            if items:
                p1s = [it['pass1'] for it in items]
                print(f'  {ds}: {len(items)} items  '
                      f'pass@1 range [{min(p1s):.4f}, {max(p1s):.4f}]  '
                      f'mean {sum(p1s)/len(p1s):.4f}')
            else:
                print(f'  {ds}: 0 items')


if __name__ == '__main__':
    main()
