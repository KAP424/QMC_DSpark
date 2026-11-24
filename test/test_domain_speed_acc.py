'''countdown + 1d-arc 上：HF 逐 token vs DSpark 多分支引擎（branches=8, stratified）的加速比与正确率。
纯 T=0.7（与 quickstart §5.3 正确率口径一致）。两个 drafter：v2-B（域扩前）与 dspark_domain（域扩后）。'''
import json, re, sys, time
from fractions import Fraction

import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_paper import DSparkDrafter, ND  # noqa: E402
import dspark_fast  # noqa: E402

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV = 'cuda:0'
N = 20
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)

items = torch.load('/home/dev/KAP/data/domain_exp/prompts.pt')
val = [c for c in items if c['split'] == 'val']
cd = [c for c in val if c['dataset'] == 'countdown'][:N]
arc = [c for c in val if c['dataset'] == '1d_arc'][:N]

def countdown_truth(nums, target):
    def gen(items):
        if len(items) == 1:
            yield items[0]; return
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                x, sx = items[i]; y, sy = items[j]
                others = [items[k] for k in range(n) if k != i and k != j]
                for v, s in ((x + y, f'({sx}+{sy})'), (x - y, f'({sx}-{sy})'),
                             (x * y, f'({sx}*{sy})')) + (((x / y, f'({sx}/{sy})'),) if y != 0 else ()):
                    yield from gen(others + [(v, s)])
    for v, s in gen([(Fraction(n_), str(n_)) for n_ in nums]):
        if v == target:
            return s
    return None

def check_countdown(resp, nums, target):
    cands = re.findall(r'[\d\(\)\+\-\*\/][\d\(\)\+\-\*\/\s]*[\d\)]', resp)
    for c in cands:
        if not re.search(r'[\+\-\*\/]', c): continue
        runs = re.findall(r'\d+', c)
        if sorted(map(int, runs)) != sorted(nums): continue
        try: val = eval(c, {'__builtins__': {}}, {})
        except Exception: continue
        if isinstance(val, float) and abs(val - round(val)) < 1e-9: val = round(val)
        if val == target: return True
    return False

def check_arc(resp, test_out):
    truth = test_out[0]; width = len(truth)
    for line in resp.splitlines():
        xs = re.findall(r'\d+', line)
        if len(xs) == width:
            return list(map(int, xs)) == truth
    return False

def check(it, resp):
    if it['dataset'] == 'countdown':
        return check_countdown(resp, it['raw']['nums'], it['raw']['target'])
    return check_arc(resp, it['raw']['test']['output'])

def hf_gen(ids, max_new_tokens=512):
    t = torch.tensor([ids], device=DEV)
    with torch.inference_mode():
        out = model.generate(input_ids=t, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, pad_token_id=tok.eos_token_id)
    n = int(out.shape[1] - t.shape[1])
    return tok.decode(out[0][t.shape[1]:], skip_special_tokens=True), n

def load_drafter(ckpt):
    d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
    c = torch.load(ckpt, map_location=DEV, weights_only=False)
    d.load_state_dict(c['model']); d.eval()
    return d, c.get('tau')

results = {}
for ds_name, ds in (('countdown', cd), ('1d_arc', arc)):
    r = {}
    ts, ntok, acc = [], [], 0
    for it in ds:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        resp, n = hf_gen(it['ids'])
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0); ntok.append(n)
        acc += check(it, resp)
    r['hf'] = {'tok_s': sum(ntok) / sum(ts), 'acc': acc}
    print(f'{ds_name}: HF {r["hf"]["tok_s"]:.1f} tok/s  acc {acc}/{N}', flush=True)
    for name, ckpt in (('v2B', '/home/dev/KAP/parameter/draft/dspark_pb_v2_b/best.pt'),
                       ('domain', '/home/dev/KAP/parameter/draft/dspark_domain/best.pt')):
        d, tref = load_drafter(ckpt)
        eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=0.7, seed=11)
        spd, acc = [], 0
        for it in ds:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids_out, st = eng.generate(it['ids'], 512, branches=8, mode='stratified')
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            spd.append(st['tokens'] / dt)
            acc += check(it, tok.decode(ids_out, skip_special_tokens=True))
        r[name] = {'tok_s': sum(spd) / len(spd), 'acc': acc, 'speedup': r['hf']['tok_s'] / (sum(spd) / len(spd)),
                   'drafter_tau': tref}
        print(f'{ds_name}: DSpark[{name}] {r[name]["tok_s"]:.1f} tok/s  speedup {r[name]["speedup"]:.2f}x  '
              f'acc {acc}/{N}', flush=True)
        del d, eng
        torch.cuda.empty_cache()
    results[ds_name] = r

json.dump(results, open('/home/dev/KAP/data/domain_exp/speed_acc.json', 'w'), indent=1)
print('\n=== 汇总（纯 T=0.7，branches=8 stratified）===')
for ds_name, r in results.items():
    print(f"[{ds_name}] HF: {r['hf']['tok_s']:.1f} tok/s, 正确率 {r['hf']['acc']}/{N}")
    for name in ('v2B', 'domain'):
        print(f"  DSpark[{name}]: {r[name]['tok_s']:.1f} tok/s  加速比 {r[name]['speedup']:.2f}x  正确率 {r[name]['acc']}/{N}")
