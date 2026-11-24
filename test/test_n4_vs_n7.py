"""Compare n=4-trained drafter vs n=7-trained drafter.
n=4 model: inference at n=4 (its native slot count).
n=7 model: inference at n=6 (peak from previous sweep).
Eval on PB val + maze + sudoku, B=4 branches, 50 prompts each.

Usage:
  python -I test_n4_vs_n7.py --ckpt4 .../dspark_n4/best.pt
                              --ckpt7 .../dspark_maze_sudoku/best.pt
"""
import argparse, json, sys, time
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
import dspark_fast

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt4', required=True)
ap.add_argument('--ckpt7', required=True)
ap.add_argument('--n-samples', type=int, default=50)
a = ap.parse_args()

P   = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV = 'cuda:0'

tok   = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

DATASETS = {
    'pb':        ('/home/dev/KAP/data/pb_dspark/prompts.pt', 'val', None),
    'maze':      ('/home/dev/KAP/data/maze_sudoku/prompts.pt', 'val', 'maze'),
    'sudoku':    ('/home/dev/KAP/data/maze_sudoku/prompts.pt', 'val', 'sudoku'),
    'countdown': ('/home/dev/KAP/data/countdown/prompts.pt', 'val', None),
}

def load_prompts(path, split, dataset_filter, n):
    items = torch.load(path, map_location='cpu', weights_only=False)
    out = []
    for it in items:
        if it['split'] != split:
            continue
        if dataset_filter and it.get('dataset') != dataset_filter:
            continue
        out.append(it['ids'])
        if len(out) >= n:
            break
    return out

def eval_drafter(ckpt_path, n_draft, infer_n, label):
    d = DSparkDrafter(model, n_draft=n_draft, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
    c = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    sd = c['model']
    if sd['mask_emb'].shape[0] != n_draft:
        sd['mask_emb'] = sd['mask_emb'][:n_draft]
    d.load_state_dict(sd)
    d.eval()
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42)

    results = {}
    for ds_name, (path, split, ds_filter) in DATASETS.items():
        try:
            prompts = load_prompts(path, split, ds_filter, a.n_samples)
        except FileNotFoundError:
            print(f'  {ds_name}: prompts.pt not found, skipping', flush=True)
            continue
        if not prompts:
            print(f'  {ds_name}: no prompts found, skipping', flush=True)
            continue
        taus, toks_per_s = [], []
        for pr in prompts:
            t0 = time.time()
            _, st = eng.generate(pr, 256, n=infer_n, branches=4, mode='stratified')
            dt = time.time() - t0
            tau = st['tokens'] / st['rounds']
            taus.append(tau)
            toks_per_s.append(st['tokens'] / dt)
        mean_tau = sum(taus) / len(taus)
        mean_tps = sum(toks_per_s) / len(toks_per_s)
        results[ds_name] = {'tau': round(mean_tau, 3), 'tok_s': round(mean_tps, 1), 'n': len(taus)}
        print(f'  {label}  n_draft={n_draft} infer_n={infer_n}  {ds_name:10s}  tau={mean_tau:.3f}  tok/s={mean_tps:.1f}', flush=True)
    del d, eng
    return results

print('=== n=4 model (inference n=4) ===', flush=True)
r4 = eval_drafter(a.ckpt4, n_draft=4, infer_n=4, label='n4')

print('=== n=7 model (inference n=6) ===', flush=True)
r7 = eval_drafter(a.ckpt7, n_draft=7, infer_n=6, label='n7')

print('\n=== SUMMARY ===', flush=True)
print(f'{"dataset":12s}  {"n4 tau":>8s}  {"n7@6 tau":>8s}  {"delta":>8s}  {"n4 tok/s":>10s}  {"n7@6 tok/s":>10s}')
for ds in r4:
    if ds not in r7:
        continue
    t4, t7 = r4[ds]['tau'], r7[ds]['tau']
    s4, s7 = r4[ds]['tok_s'], r7[ds]['tok_s']
    print(f'{ds:12s}  {t4:8.3f}  {t7:8.3f}  {t4-t7:+8.3f}  {s4:10.1f}  {s7:10.1f}')

out = {'n4': r4, 'n7_at_6': r7}
with open('/home/dev/KAP/parameter/draft/dspark_n4/compare_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nresults written to dspark_n4/compare_results.json', flush=True)
