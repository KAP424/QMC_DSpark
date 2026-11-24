"""Measure DSpark vs HF-baseline speedup on maze and sudoku tasks.
Uses dspark_maze_sudoku/best.pt (trained on PB + maze + sudoku).
Outputs tok/s, speedup, and tau_real for each domain.
"""
import sys, time, torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
import dspark_fast

P    = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV  = 'cuda:0'
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='/home/dev/KAP/parameter/draft/dspark_maze_sudoku/best.pt')
_ap.add_argument('--out', default='/home/dev/KAP/data/maze_sudoku/speedup_results.json')
_args = _ap.parse_args()
CKPT = _args.ckpt
OUT_PATH = _args.out
N_PROMPTS = 32
MAX_NEW   = 256

tok   = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(
    P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()
eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42)

items_ms = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt',
                      map_location='cpu', weights_only=False)

def load_val(dataset, n):
    return [it['ids'] for it in items_ms
            if it['split'] == 'val' and it['dataset'] == dataset][:n]

@torch.inference_mode()
def baseline_one(prompt_ids):
    t0 = time.perf_counter()
    ids = torch.tensor([prompt_ids], device=DEV)
    gen = model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                         do_sample=True, temperature=1.0,
                         pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    n_new = gen.shape[1] - ids.shape[1]
    return n_new, dt

def dspark_one(prompt_ids, branches=4):
    t0 = time.perf_counter()
    out, st = eng.generate(prompt_ids, MAX_NEW, branches=branches, mode='stratified')
    dt = time.perf_counter() - t0
    return st['tokens'], dt, st['tokens'] / st['rounds']

results = {}
for ds in ('maze', 'sudoku'):
    prompts = load_val(ds, N_PROMPTS)
    if not prompts:
        print(f'{ds}: no val prompts found'); continue
    print(f'\n=== {ds} ({len(prompts)} prompts) ===')

    # warmup
    baseline_one(prompts[0])
    dspark_one(prompts[0])

    base_tps, ds_tps, taus = [], [], []
    for pr in prompts:
        n, dt = baseline_one(pr)
        base_tps.append(n / dt)

        n2, dt2, tau = dspark_one(pr)
        ds_tps.append(n2 / dt2)
        taus.append(tau)

    mb = sum(base_tps) / len(base_tps)
    md = sum(ds_tps)   / len(ds_tps)
    mt = sum(taus)     / len(taus)
    sp = md / mb
    results[ds] = dict(baseline_tps=round(mb,1), dspark_tps=round(md,1),
                       speedup=round(sp,3), tau=round(mt,3))
    print(f'  baseline:  {mb:.1f} tok/s')
    print(f'  DSpark B=4: {md:.1f} tok/s   τ={mt:.3f}')
    print(f'  speedup:   {sp:.2f}x')

import json, os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
json.dump(results, open(OUT_PATH, 'w'), indent=2)
print(f'\nresults saved to {OUT_PATH}')
print('\nSUMMARY')
print(f"{'domain':10s}  {'baseline tok/s':>14s}  {'DSpark tok/s':>12s}  {'speedup':>8s}  {'tau':>6s}")
for ds, r in results.items():
    print(f"{ds:10s}  {r['baseline_tps']:14.1f}  {r['dspark_tps']:12.1f}  {r['speedup']:7.2f}x  {r['tau']:6.3f}")
