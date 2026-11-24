import json

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

W = '/home/dev/KAP/data/pb_dspark'
PAR = '/home/dev/KAP/parameter/draft'
PY = '/home/dev/venv/bin/python'
TARGET = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'

cells = []

cells.append(md(f'''# DSpark v2 训练 · 运行与监控

**训练代码就是 [`train/pb_train_v2.py`](pb_train_v2.py)**（官方 DeepSpec 配方；旧的 `pb_train.py` 是第一版 1.806 用的，保留作对照）。三个脚本的关系：

| 文件 | 作用 |
| --- | --- |
| [`pb_gen_worker.py`](pb_gen_worker.py) | ① 数据生成：target 以 **T=0.7 / top_p 0.8 / top_k 20** 给每条 prompt 生成 response（可断点续跑，按 id 跳过已完成） |
| [`pb_train_v2.py`](pb_train_v2.py) | ② 训练：γ=4 加权平均 loss（0.1·CE + 0.9·L1 + 1.0·BCE）、AdamW 6e-4 + cosine(4% warmup)、wd 0、bf16、每序列多 anchor；每 1000 步在 200 条 val 上评 τ 并存 `best.pt`/`last.pt` |
| [`pb_pipeline_v2.sh`](pb_pipeline_v2.sh) | 一键流水线：等生成完 → 校验齐全 → 显存探针 → 自动启动两个训练 |

输入 / 输出：

| 路径 | 内容 |
| --- | --- |
| `{W}/prompts.pt` | 50,500 条 prompt（50,000 train + 500 val，已切分） |
| `{W}/gen07/` | 0.7/0.8/20 生成的 response —— **训练数据** |
| `{W}/gen/` | T=1.0 生成的 response —— 论文口径评测（τ_paper）用 |
| `{PAR}/dspark_pb_v2_a/` | 运行 A 权重（`best.pt`/`last.pt`）+ `train_log.jsonl` |
| `{PAR}/dspark_pb_v2_b/` | 运行 B 权重 + 日志 |

**当前状态（2026-11-14）**：A 已完成（30,860 步，τ_paper **2.133**，第一版是 1.806）；B 进行中（τ_paper 2.137 @ 47k/61,710 步）。完整差距分析和配方依据见 [`doc/DSpark-训练文档.md`](../doc/DSpark-训练文档.md) §12。'''))

cells.append(md('## 0. 状态检查（可随时运行）'))

cells.append(code(f'''!nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
!pgrep -af pb_train_v2.py || echo "(没有训练进程在运行)"
print('--- 运行 A 日志末尾 ---')
!tail -n 2 {W}/train_v2_a.log
print('--- 运行 B 日志末尾 ---')
!tail -n 2 {W}/train_v2_b.log'''))

cells.append(md(f'''## 1. 数据生成（已完成，重跑会自动跳过）

gen07/ 里已有 50,500 条（2026-11-14 00:29 校验齐全）。**只有换模型、换采样参数或换 max_new 时才需要重新生成**；同参数重跑安全（按 id 断点续跑），但会占满两张卡约 2.5 小时。

```
cd /home/dev/KAP/train
nohup env CUDA_VISIBLE_DEVICES=0 {PY} -I pb_gen_worker.py --rank 0 --world 2 \\
    --work {W} --target {TARGET} \\
    --max-new 1024 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 96 --out gen07 \\
    > {W}/gen07/log_r0.txt 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 {PY} -I pb_gen_worker.py --rank 1 --world 2 \\
    --work {W} --target {TARGET} \\
    --max-new 1024 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 96 --out gen07 \\
    > {W}/gen07/log_r1.txt 2>&1 &
```

关键参数：`--max-new 1024`（官方 4096 的折中）、`--temperature/--top-p/--top-k`（官方推荐采样参数）、`--out gen07`（与旧 T=1.0 数据分开存）。'''))

cells.append(md('## 2. 训练（pb_train_v2.py）'))

cells.append(md(f'''**参数表**：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--anchors` | 32 | 每序列采样的 anchor 数（官方 512；这是 v2 相对 v1 最大的改动之一） |
| `--seqs` | 16 | 每步序列数；blocks/步 = anchors × seqs（32×16 = 512，实测 28.6 GB） |
| `--epochs` | 10 | 官方同款 |
| `--lr` | 6e-4 | 官方 |
| `--warmup` | 0.04 | 线性 warmup 后 cosine 到 0（官方） |
| `--gamma` | 4.0 | loss 位置衰减 w_k = exp(−(k−1)/γ)（官方 4，v1 误用 7） |
| `--eval-every` | 1000 | 每 N 步评一次 τ 并存 best/last |
| `--out` | 必填 | 权重与日志目录 |
| `--resume` | 开关 | 从 `last.pt` 断点续训（调度按总步数对齐） |
| `--probe` | 开关 | 只跑 3 步，打印峰值显存和每步耗时后退出 |

**两个运行的配置**（当时两张卡各跑一个）：

- 运行 A：`--anchors 32 --seqs 16` → 512 blocks/步，49,376 序列 × 10 epochs = 30,860 步
- 运行 B：`--anchors 64 --seqs 8` → 同样 512 blocks/步，但每序列覆盖翻倍，61,710 步

**启动前必须确认两张卡空闲**（上面 §0 的 nvidia-smi）。启动命令（去掉行首 `#` 才会执行）：'''))

cells.append(code(f'''cd /home/dev/KAP/train
# nohup env CUDA_VISIBLE_DEVICES=0 {PY} -I pb_train_v2.py --anchors 32 --seqs 16 --epochs 10 \\
#     --out {PAR}/dspark_pb_v2_a > {W}/train_v2_a.log 2>&1 &
# nohup env CUDA_VISIBLE_DEVICES=1 {PY} -I pb_train_v2.py --anchors 64 --seqs 8 --epochs 10 \\
#     --out {PAR}/dspark_pb_v2_b > {W}/train_v2_b.log 2>&1 &
print('启动命令已注释：先跑 §0 确认 GPU 空闲，再去掉 # 执行；断点续训在命令末尾加 --resume')'''))

cells.append(md('''## 3. 训练曲线（只读文件，随时可运行）'''))

cells.append(code('''import json
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'          # 已通过色觉/对比度校验的分类色
INK, MUTE, GRID = '#3d3d3a', '#6f6e68', '#e3e2dc'
OLD_TAU = 1.806                               # 第一版训练（同评测口径）的 tau_paper

def read(tag):
    try:
        return [json.loads(l) for l in open(f'/home/dev/KAP/parameter/draft/dspark_pb_v2_{tag}/train_log.jsonl')]
    except FileNotFoundError:
        return []

A, B = read('a'), read('b')
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
panels = [('tau_paper', 'tau_paper - T=1.0 responses, T=1.0 eval (paper protocol)'),
          ('tau_indist', 'tau_indist - 0.7/0.8/20 responses (training distribution)')]
for ax, (key, title) in zip(axes, panels):
    for recs, name, color, ls in ((A, 'Run A: 32 anchors x 16 seqs', BLUE, '-'),
                                  (B, 'Run B: 64 anchors x 8 seqs', ORANGE, '--')):
        if recs:
            ax.plot([r['step'] for r in recs], [r[key] for r in recs], color=color, lw=2, ls=ls, label=name)
    ax.set_xlabel('step'); ax.set_ylabel(key)
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b0')
    ax.tick_params(colors=MUTE, labelsize=9)
if A:
    axes[0].axhline(OLD_TAU, color='#9a9992', lw=1.2, ls=':')
    axes[0].annotate(f'v1 tau={OLD_TAU}', xy=(0.98, OLD_TAU), xycoords=('axes fraction', 'data'),
                     ha='right', va='bottom', fontsize=8, color=MUTE)
axes[0].legend(frameon=False, fontsize=9, loc='lower right')
fig.suptitle('DSpark v2 training curves (official DeepSpec recipe)', color=INK, fontsize=12, x=0.01, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()'''))

cells.append(md('''### 3.1 接受率 α 随训练的变化（每个草稿位置）

αₖ = Σ min(pₖ, qₖ) 是第 k 个草稿位置在 T=1.0 随机接受下的接受概率（τ_paper 口径，每 1000 步在 200 条 val 上评一次）。7 条线用同一色相由浅到深表示位置 1→7：**训练好不好，看的是所有位置的 α 是否整体上移**（第一版最终 α₁ 只有 0.50，v2 到 0.61）。'''))

cells.append(code('''import json
import matplotlib.pyplot as plt

INK, MUTE, GRID = '#3d3d3a', '#6f6e68', '#e3e2dc'
RAMP = ['#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#104281', '#0d366b']  # slot 1..7, one hue light->dark

def read(tag):
    try:
        return [json.loads(l) for l in open(f'/home/dev/KAP/parameter/draft/dspark_pb_v2_{tag}/train_log.jsonl')]
    except FileNotFoundError:
        return []

A, B = read('a'), read('b')
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110, sharey=True)
for ax, recs, title in ((axes[0], A, 'Run A: 32 anchors x 16 seqs'), (axes[1], B, 'Run B: 64 anchors x 8 seqs')):
    for k in range(7):
        ax.plot([r['step'] for r in recs], [r['alpha_paper'][k] for r in recs],
                color=RAMP[k], lw=1.8, label=f'slot {k+1}')
    ax.set_xlabel('step'); ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b0')
    ax.tick_params(colors=MUTE, labelsize=9)
axes[0].set_ylabel('acceptance rate alpha_k')
axes[1].legend(frameon=False, fontsize=8, ncol=2, loc='lower right')
fig.suptitle('Per-position acceptance rate (tau_paper protocol, T=1.0)', color=INK, fontsize=12, x=0.01, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()'''))

cells.append(md('''### 3.2 loss 下降曲线

**注意：v2 的两个训练启动时，脚本还没有记录每步 loss（我的疏漏，2026-11-14 已在 `pb_train_v2.py` 修复）**——所以 A 只有 τ/α 曲线，没有 loss 曲线。修复后启动的训练会自动把每 50 步的 loss/ce/l1/bce 写进 `<out>/loss_log.jsonl`，下面的 cell 会自动画出来。**loss 绝对值只在同一套定义内可比**（v1 位置求和 γ=7，v2 加权平均 γ=4，不能互比），每个运行看自己的下降趋势即可。'''))

cells.append(code('''import json
import matplotlib.pyplot as plt

INK, MUTE, GRID = '#3d3d3a', '#6f6e68', '#e3e2dc'
BLUE = '#2a78d6'

SRC = [
    ('v1 run: 8 anchors, T=1.0 data (loss: summed, gamma=7)',
     '/home/dev/KAP/data/pb_dspark/train_log.jsonl'),
    ('v2 Run A: 32 anchors, official recipe (loss: weighted mean, gamma=4)',
     '/home/dev/KAP/parameter/draft/dspark_pb_v2_a/loss_log.jsonl'),
    ('v2 Run B: 64 anchors, official recipe (loss: weighted mean, gamma=4)',
     '/home/dev/KAP/parameter/draft/dspark_pb_v2_b/loss_log.jsonl'),
]

have = []
for name, path in SRC:
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        continue
    tr = [r for r in rows if r.get('t') == 'train' or r.get('type') == 'train']
    if tr:
        have.append((name, tr))

if not have:
    print('目前没有 loss 记录：v2 的 A/B 启动时脚本尚未记录 loss（已修复，下次启动的训练自动写入 <out>/loss_log.jsonl）。')
else:
    fig, axes = plt.subplots(1, len(have), figsize=(5.5 * len(have), 3.8), dpi=110)
    axes = [axes] if len(have) == 1 else list(axes)
    for ax, (name, tr) in zip(axes, have):
        ax.plot([r['step'] for r in tr], [r['loss'] for r in tr], color=BLUE, lw=2)
        ax.set_xlabel('step'); ax.set_ylabel('loss')
        ax.set_title(name, fontsize=9.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b0')
        ax.tick_params(colors=MUTE, labelsize=9)
    fig.tight_layout()
    plt.show()'''))

cells.append(md('''## 4. 评测一个 checkpoint（τ 复测）

加载 `best.pt`，在与训练完全相同的 200 条 val 上复算两个口径的 τ。**默认用 cuda:0**（运行 A 已结束、该卡空闲）；如果两张卡都在训练，等空闲再跑。需要约 4 GB 显存、约 1 分钟。'''))

cells.append(code(f'''import glob, math, sys
import torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
from dspark_loss import make_blocks

P = '{TARGET}'
W = '{W}'
CKPT = '{PAR}/dspark_pb_v2_a/best.pt'      # 想评别的权重就改这里，如 dspark_pb_v2_b/last.pt
DEV = 'cuda:0'

tok = AutoTokenizer.from_pretrained(P); EOS = tok.eos_token_id
items = {{c['id']: c for c in torch.load(W + '/prompts.pt')}}

def load_resp(d):
    r = {{}}
    for f in sorted(glob.glob(f'{{W}}/{{d}}/shard_r*_*.pt')):
        for x in torch.load(f):
            r[x['id']] = x['resp']
    return r

gen10, gen07 = load_resp('gen'), load_resp('gen07')
val10 = [{{'id': i, 'prompt_ids': c['ids'], 'resp_ids': gen10[i]}}
         for i, c in items.items() if c['split'] == 'val' and i in gen10 and len(gen10[i]) >= 16][:200]
val07 = [{{'id': i, 'prompt_ids': c['ids'], 'resp_ids': gen07[i]}}
         for i, c in items.items() if c['split'] == 'val' and i in gen07 and len(gen07[i]) >= 16][:200]

model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()

@torch.inference_mode()
def tau(seqs, A=8, bs=16, seed=123):          # 与训练时 eval_tau 完全同协议
    g = torch.Generator().manual_seed(seed)
    al, n = None, 0
    for b0 in range(0, len(seqs), bs):
        feats, toks, o, p_t = make_blocks(model, EOS, seqs[b0:b0 + bs], A, g, 1.0, DEV)
        with torch.autocast('cuda', torch.bfloat16):
            z = d.backbone(d.encode_ctx(feats), d.embed(toks[..., 0]), o)
            lg = d.logits(z, toks[..., :ND]).float()
        q = torch.softmax(lg, -1)
        a = torch.minimum(p_t, q).sum(-1).mean((0, 1)).cpu()
        al = a if al is None else al + a
        n += 1
    al = (al / n).tolist()
    en = sum(math.prod(al[:k + 1]) for k in range(len(al)))
    return en + 1.0, al

for name, vs in (('τ_paper  (T=1.0 生成的 response)', val10), ('τ_indist (0.7/0.8/20 生成的 response)', val07)):
    t, al = tau(vs)
    a_str = ' '.join(f'{{x:.2f}}' for x in al)
    print(f'{{name}}: τ = {{t:.3f}}   α = {{a_str}}')
print(f"checkpoint: {{CKPT}}  step {{ck.get('step')}}  训练时记录 τ={{ck.get('tau')}}")'''))

cells.append(md('''## 5. 常用操作备忘

```bash
# 断点续训（在原命令后加 --resume，其余不变）
... pb_train_v2.py --anchors 32 --seqs 16 --epochs 10 --out /home/dev/KAP/parameter/draft/dspark_pb_v2_a --resume

# 停掉训练（会丢未存盘的进度；每 1000 步存一次 last.pt，损失最多 1000 步）
pkill -f pb_train_v2.py

# 看实时日志
tail -f /home/dev/KAP/data/pb_dspark/train_v2_a.log

# 校验 gen07 数据是否齐全（每条 prompt 都有 response）
/home/dev/venv/bin/python -I /home/dev/KAP/train/pb_check_gen.py   # 旧数据；gen07 同理改目录
```

- **τ 怎么读**：τ_paper = 每轮平均接受 token 数（含 bonus），T=1.0 口径，与论文表 1 同定义；第一版 1.806 → v2 2.13+。τ_indist 是训练同分布（0.7 采样）上的值，只用于看拟合程度，不与论文直接比。
- 权重目录里 `best.pt` 按 τ_paper 最优保存，`last.pt` 是最近一次评测的完整恢复点（含优化器/调度状态，供 --resume）。
- 论文对齐情况与剩余差距（数据量、max_new、多轮）见 `doc/DSpark-训练文档.md` §12。'''))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'venv', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.10'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('/home/dev/KAP/train/dspark_v2_run.ipynb', 'w'), indent=1, ensure_ascii=False)
print('written', sum(1 for c in cells), 'cells')
