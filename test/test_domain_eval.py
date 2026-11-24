'''Domain evaluation: tau_real + tok/s of the multi-branch engine (branches=8, stratified, T=1.0)
on three domains: countdown val, 1d-arc val, open-perfectblend val. Writes JSON.
Usage: python -I test_domain_eval.py --ckpt <best.pt> --out <json>'''
import argparse, json, sys, time

import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_paper import DSparkDrafter, ND  # noqa: E402
import dspark_fast  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--branches', type=int, default=8)
ap.add_argument('--max-new', type=int, default=256)
a = ap.parse_args()
DEV = 'cuda:0'
P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(a.ckpt, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()
eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=11)

items = torch.load('/home/dev/KAP/data/domain_exp/prompts.pt')
dom = [c['ids'] for c in items if c['split'] == 'val' and c['dataset'] == 'countdown'][:16]
arc = [c['ids'] for c in items if c['split'] == 'val' and c['dataset'] == '1d_arc'][:16]
pb = [c['ids'] for c in torch.load('/home/dev/KAP/data/pb_dspark/prompts.pt') if c['split'] == 'val'][:12]

results = {'ckpt': a.ckpt, 'ckpt_tau': float(ck.get('tau', 0)), 'branches': a.branches}
for name, plist in (('countdown', dom), ('1d_arc', arc), ('pb', pb)):
    ts, taus = [], []
    for pr in plist:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, st = eng.generate(pr, a.max_new, branches=a.branches, mode='stratified')
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        ts.append(st['tokens'] / dt); taus.append(st['tokens'] / st['rounds'])
    results[name] = {'tok_s': sum(ts) / len(ts), 'tau': sum(taus) / len(taus), 'n': len(plist)}
    print(f"{name:10s}: {results[name]['tok_s']:6.1f} tok/s  tau {results[name]['tau']:.2f}", flush=True)
json.dump(results, open(a.out, 'w'), indent=1)
print('saved', a.out)
