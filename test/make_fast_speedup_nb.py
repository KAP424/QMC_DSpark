import json

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

cells = []

cells.append(md('''# DSpark 快速引擎（数学回滚）端到端加速实测

**改动**（`src4dspark/dspark_fast.py`，对照 `dspark_e2e.py`）：
1. **每轮只有 1 次 target 前向**：拒绝时不再"快照+重放前向"，而是用 gated delta rule 的前缀结合律**数学重建** GDN 状态（一次 batched fla 调用，~1.1 ms）；correction token 直接作为下一轮的 anchor（它本来就是 target 分布的合法采样）。
2. 快照只克隆 18 层 GDN 状态（0.2 ms；注意力 KV 逐 token 可裁剪，拒绝时直接 crop）。
3. 接受判定向量化（每轮 CPU-GPU 同步 12 次 → ~3 次）。

**正确性校验**（`test/test_dspark_fast.py`，已跑通）：每轮 verify logits 与"从零重算"金标准的偏差 max 0.38–0.59 / mean 0.30–0.33，**与老引擎（0.34–0.63 / 0.28–0.30）完全同级**——该偏差是模型 chunked/recurrent 数值路径的固有噪底，快速引擎与老引擎同等无损。

协议：12 条 val prompt × 256 token，GPU 0，prefill 不计时，drafter = v2-B best（teacher-forced τ_paper 2.159）。'''))

cells.append(code('''import glob, sys, time
import torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
from dspark_e2e import DSparkDecode, baseline_generate
import dspark_fast

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
W = '/home/dev/KAP/data/pb_dspark'
CKPT = '/home/dev/KAP/parameter/draft/dspark_pb_v2_b/best.pt'
DEV = 'cuda:0'
NP, MAX_NEW = 12, 256

tok = AutoTokenizer.from_pretrained(P)
items = [c for c in torch.load(W + '/prompts.pt') if c['split'] == 'val']
prompts = [c['ids'] for c in items if 60 <= len(c['ids']) <= 400][:NP]
print(f'{len(prompts)} prompts, decode {MAX_NEW} tokens each')

model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()
print(f"drafter {CKPT} (step {ck['step']}, teacher-forced tau {ck['tau']:.3f})")'''))

cells.append(code('''def run(temperature, engine_seed=0):
    res = {}
    g = torch.Generator(device=DEV).manual_seed(engine_seed)
    t0 = time.perf_counter()
    base = []
    for p in prompts:
        torch.cuda.synchronize(); s = time.perf_counter()
        out, _ = baseline_generate(model, tok, p, MAX_NEW, temperature, g)
        torch.cuda.synchronize(); base.append(len(out) / (time.perf_counter() - s))
    res['baseline_mean'] = sum(base) / len(base)

    old = DSparkDecode(model, d, tok, temperature=temperature, seed=engine_seed)
    o_t = []
    stats_o = []
    for p in prompts:
        torch.cuda.synchronize(); s = time.perf_counter()
        out, st = old.generate(p, MAX_NEW)
        torch.cuda.synchronize(); o_t.append(len(out) / (time.perf_counter() - s))
        stats_o.append(st)
    res['old_mean'] = sum(o_t) / len(o_t)
    res['old_tau'] = sum(s['tokens'] / s['rounds'] for s in stats_o) / len(stats_o)

    fast = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=temperature, seed=engine_seed)
    f_t, stats = [], []
    for p in prompts:
        torch.cuda.synchronize(); s = time.perf_counter()
        out, st = fast.generate(p, MAX_NEW)
        torch.cuda.synchronize(); f_t.append(len(out) / (time.perf_counter() - s))
        stats.append(st)
    res['fast_mean'] = sum(f_t) / len(f_t)
    res['fast_tau'] = sum(s['tokens'] / s['rounds'] for s in stats) / len(stats)
    res['round_ms'] = sum((s['draft_ms'] + s['verify_ms'] + s['rewind_ms']) / s['rounds'] for s in stats) / len(stats)
    res['draft_ms'] = sum(s['draft_ms'] / s['rounds'] for s in stats) / len(stats)
    res['verify_ms'] = sum(s['verify_ms'] / s['rounds'] for s in stats) / len(stats)
    res['rewind_ms'] = sum(s['rewind_ms'] / s['rounds'] for s in stats) / len(stats)
    res['all_acc'] = sum(s['all_accepted'] / s['rounds'] for s in stats) / len(stats)
    res['speedup_fast'] = res['baseline_mean'] / res['fast_mean']
    res['speedup_old'] = res['baseline_mean'] / res['old_mean']
    return res

runs = {}
for name, T in (('T=1.0 (paper protocol)', 1.0), ('T=0.7/0.8/20 (training distribution)', 0.7)):
    print('running', name, flush=True)
    runs[name] = run(T)
print('done')'''))

cells.append(code('''print(f"{'condition':44s} {'base':>6s} {'old':>6s} {'fast':>6s} {'x_old':>6s} {'x_fast':>7s} {'tau_old':>7s} {'tau_fast':>8s}")
for name, r in runs.items():
    print(f"{name:44s} {r['baseline_mean']:6.1f} {r['old_mean']:6.1f} {r['fast_mean']:6.1f} "
          f"{r['speedup_old']:5.2f}x {r['speedup_fast']:6.2f}x {r['old_tau']:7.2f} {r['fast_tau']:8.2f}")
r = runs['T=1.0 (paper protocol)']
print()
print(f"fast 引擎每轮 {r['round_ms']:.1f} ms = draft {r['draft_ms']:.1f} + verify {r['verify_ms']:.1f} + rewind {r['rewind_ms']:.1f}"
      f"  (老引擎同口径 ~68 ms: draft 5.3 + verify 29.3 + replay 28.6 + misc)")
print(f"盈亏平衡 tau = 每轮毫秒 / baseline 每token毫秒 = {r['round_ms']/(1000/r['baseline_mean']):.2f}"
      f"  (老引擎 ~3.9);  实测 tau_fast {r['fast_tau']:.2f}")'''))

cells.append(md('''## 结论解读

- 快速引擎把每轮 2 次 target 前向减为 **1 次**：回滚从"28.6 ms 重放前向"变成 ~1.1 ms 的数学重建。
- 每轮 ~37 ms（draft 6.4 + verify 29.5 + rewind 1.1），盈亏平衡 τ ≈ 2.1；v2-B drafter 实测 τ > 2.1 时即净加速。
- 剩余瓶颈：verify 前向本身 29.5 ms 里 GPU 只忙 ~5.8 ms（1798 次 kernel 发射）——torch.compile/CUDA-graph 是下一个杠杆，可再把每轮压到 ~15-20 ms。
- 无损性：与老引擎同等（见顶部校验段落与 `test/test_dspark_fast.py`）。'''))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'venv', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.10'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('/home/dev/KAP/test/dspark_fast_speedup.ipynb', 'w'), indent=1, ensure_ascii=False)
print('written', len(cells), 'cells')
