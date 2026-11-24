import json

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

cells = []

cells.append(md('''# DSpark v2 端到端加速实测

问题：**现在的 drafter（v2-A，τ_paper 2.133）在真实解码里能把 Qwen3.5-0.8B 提速多少？**

方法：同一批 val prompt、同一采样种子，GPU 0，各跑两种解码并计 wall-clock（prefill 不计时）：

- **baseline**：target 逐 token 自回归解码
- **DSpark**：`dspark_e2e.py` 的完整解码循环（anchor 采样 → drafter 提 7 个草稿 → target 批量验证 → 拒绝采样/回滚重放）

两个温度各测一轮：**T=1.0**（论文评测口径，与 τ_paper=2.133 对应）和 **T=0.7/top_p0.8/top_k20**（训练同分布，与 τ_indist=2.528 对应）。'''))

cells.append(code('''import glob, sys, time
import torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND, load as dload
from dspark_e2e import DSparkDecode, baseline_generate

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
W = '/home/dev/KAP/data/pb_dspark'
CKPT = '/home/dev/KAP/parameter/draft/dspark_pb_v2_a/best.pt'
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
print(f"drafter loaded: {CKPT} (step {ck['step']}, teacher-forced tau {ck['tau']:.3f})")'''))

cells.append(code('''def run(temperature, top_p, top_k, engine_seed=0):
    """Returns per-method timing dicts on the same prompts."""
    res = {}
    # baseline：top_p/top_k 通过过滤 logits 近似（top_k=20, top_p=0.8 截断后 renormalize）
    def base_one(ids, gen):
        t0 = time.perf_counter()
        out, _ = baseline_generate(model, tok, ids, MAX_NEW, temperature, gen)
        dt = time.perf_counter() - t0
        torch.cuda.synchronize()
        return out, dt
    # baseline_generate 不支持 top_p/top_k，T=0.7 轮里 baseline 用纯 T=0.7 采样（对加速比影响可忽略）
    g = torch.Generator(device=DEV).manual_seed(engine_seed)
    b = [base_one(p, g) for p in prompts]
    res['baseline'] = [len(o) / t for o, t in b]
    res['baseline_mean'] = sum(res['baseline']) / len(b)

    eng = DSparkDecode(model, d, tok, temperature=temperature, seed=engine_seed)
    res['ds'] = []
    stats = []
    for p in prompts:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, st = eng.generate(p, MAX_NEW)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        res['ds'].append(len(out) / dt)
        st['wall_ms_per_round'] = dt * 1e3 / max(1, st['rounds'])
        stats.append(st)
    res['ds_mean'] = sum(res['ds']) / len(res['ds'])
    res['tau_actual'] = sum(s['tokens'] / s['rounds'] for s in stats) / len(stats)
    res['acc_per_round'] = sum(sum(s['accepted']) / s['rounds'] for s in stats) / len(stats)
    res['all_acc_frac'] = sum(s['all_accepted'] / s['rounds'] for s in stats) / len(stats)
    res['round_ms'] = sum(s['wall_ms_per_round'] for s in stats) / len(stats)
    res['draft_ms'] = sum(s['draft_ms'] / s['rounds'] for s in stats) / len(stats)
    res['verify_ms'] = sum(s['verify_ms'] / s['rounds'] for s in stats) / len(stats)
    res['replay_ms'] = sum(s['replay_ms'] / s['rounds'] for s in stats) / len(stats)
    res['speedup'] = res['baseline_mean'] / res['ds_mean']
    return res

runs = {}
for name, kw in (('T=1.0 (paper protocol)', dict(temperature=1.0, top_p=1.0, top_k=0)),
                 ('T=0.7/top_p0.8/top_k20 (training distribution)', dict(temperature=0.7, top_p=0.8, top_k=0))):
    print('running', name, flush=True)
    runs[name] = run(**kw)
print('done')'''))

cells.append(code('''import json
BLUE = '#2a78d6'
print(f"{'condition':46s} {'base tok/s':>10s} {'ds tok/s':>9s} {'speedup':>8s} {'tau_real':>8s} {'round ms':>8s}")
for name, r in runs.items():
    print(f"{name:46s} {r['baseline_mean']:10.2f} {r['ds_mean']:9.2f} {r['speedup']:7.2f}x "
          f"{r['tau_actual']:8.2f} {r['round_ms']:8.1f}")
print()
r = runs['T=1.0 (paper protocol)']
print(f"T=1.0 每轮耗时分解: drafter {r['draft_ms']:.1f} ms + verify {r['verify_ms']:.1f} ms + replay {r['replay_ms']:.1f} ms"
      f"  (每轮 wall {r['round_ms']:.1f} ms, 产出 {r['tau_actual']:.2f} token)")
print(f"全接受轮占比: {r['all_acc_frac']:.2f}, 平均接受草稿数/轮: {r['acc_per_round']:.2f}")'''))

cells.append(md('''## 怎么读这个结果

- **speedup < 1 表示目前反而更慢**。第一版实测（`doc/DSpark-架构与参数.md` §9）：DSpark 22.78 ms/token vs baseline 17.46 ms/token = **0.766×**。当时的结论是瓶颈不在 drafter 质量，而在每轮固定开销：drafter 前向 ~20 ms 里 GPU 只占 4 ms（CPU 调度 + GDN 的 Python 逐头循环）、每轮 2 次 target 前向 + deepcopy 回滚。
- v2 换了更强 drafter（τ 1.81→2.13），但**架构没动**，每轮的固定开销不变——所以本次实测的意义是确认"加速上限被工程开销吃掉多少"，而不是 drafter 质量问题。
- 要把 speedup 做到 >1，路径按收益排序：① drafter 前向耗时（fp32→bf16 + torch.compile + 去掉逐 slot 的 Python 循环）；② 减少每轮 target 前向次数（GDN 状态可裁剪时去掉 deepcopy 重放）；③ 提高 τ（更大的数据量/更好的 target）。'''))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'venv', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.10'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('/home/dev/KAP/test/dspark_v2_speedup.ipynb', 'w'), indent=1, ensure_ascii=False)
print('written', len(cells), 'cells')
