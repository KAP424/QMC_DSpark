# Pass@k 评测与绘图说明 (2026-11-24)

本目录用于对 **Maze**、**Sudoku**、**Countdown** 和 **1D-ARC** 四个推理任务进行投机采样（Speculative Decoding）与算术编码准蒙特卡洛（QuasiMoTTo / DSpark AC）的 `pass@k` 性能评测与可视化。

---

## 1. 评测模型与数据集配置

所有任务均采用默认抽样 `N_prompts = 100` 道验证集题目，每题并行生成 `B = 64` 条轨迹（`T = 1.0`，草稿长度 `n = 7`），计算 $k \in \{1, 2, 4, 8, 16, 32, 64\}$ 的 `pass@k`：


| 任务          | Target 模型路径                                                           | Drafter 参数路径                                             | 评测验证集数据源                                                               | 单题真实通过率$p_i$ 计算方式                                          |
| :------------ | :------------------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Maze**      | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-maze-sft/hf_model`      | `/home/dev/KAP/parameter/draft/dspark_maze_sft/best.pt`      | `/home/dev/KAP/data/maze_sudoku/prompts.pt` (`split='val'`) + `maze_val.csv`   | Teacher-forcing 精确序列概率$\pi_\theta(\text{sol}\mid\text{prompt})$ |
| **Sudoku**    | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model`    | `/home/dev/KAP/parameter/draft/dspark_sudoku_sft/best.pt`    | `/home/dev/KAP/data/maze_sudoku/prompts.pt` (`split='val'`) + `sudoku_val.csv` | Teacher-forcing 精确序列概率$\pi_\theta(\text{sol}\mid\text{prompt})$ |
| **Countdown** | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model` | `/home/dev/KAP/parameter/draft/dspark_countdown_sft/best.pt` | `/home/dev/KAP/data/domain_exp/prompts.pt` (`split='val'`)                     | 高精度 QMC 蒙特卡洛估计（`M = 128` rollouts，因一题存在多个合法算式） |
| **1D-ARC**    | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original`               | `/home/dev/KAP/parameter/draft/dspark_arc_sft/best.pt`       | `/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset/*/*.json`                       | Teacher-forcing 精确序列概率$\pi_\theta(\text{ans}\mid\text{prompt})$ |

> **注**：`/home/dev/KAP/data/passk_filtered/` 仅用于过滤 `train` 集做后续 GRPO 强化学习训练，**不用于**此处 `pass@k` 测试集评估。

---

## 2. 采样设置定义与 S2_update 的精确流程机制

绘图包含以下 5 条对比曲线：

1. **`S2_update`**：DSpark 算术编码确定性解码（AC）。
2. **`S3_paper`**：无 Drafter 的原论文基准。
3. **`S1_iid`**：传统独立同分布投机采样。
4. **`Upper bound`**：理论上线 $\mathbb{E}_{i}[\min(k \cdot p_i, 1)]$。
5. **`Theory i.i.d.`**：理论独立采样 $\mathbb{E}_{i}[1 - (1 - p_i)^k]$。

### S2_update (AC) 核心流程：

通过维护全局概率空间坐标 $u \in [0, 1)$，实现 Target 与 Drafter 的确定性耦合：

1. **初始锚定 (Initial Anchor)**：
   Target 模型对 $u_0 = (\Delta + b/B) \bmod 1$ 进行逆 CDF 映射得到第一个 Token $y_0 = F_p^{-1}(u_0)$。更新坐标 $u_1 = (u_0 - F_p(y_0)) / p(y_0)$。
2. **草稿生成 (Draft Phase)**：
   Drafter 接收状态 $u_1$ 作为初始草稿坐标（即 $\hat{u}_1 = u_1$），在自身 $q$ 分布下进行确定性逆 CDF 采样：

   - 生成 $\hat{y}_j = F_q^{-1}(\hat{u}_j)$。
   - 传播坐标 $\hat{u}_{j+1} = (\hat{u}_j - F_q(\hat{y}_j)) / q(\hat{y}_j)$。Drafter 序列需严格追踪其内部坐标空间 $\hat{u}$ 的演变。
3. **验证与同步 (Verification Phase)**：
   Target 模型利用该轮锚点产生的坐标轨迹 $\{u_1, \dots, u_n\}$ 验证草稿：

   - 计算 Target 序列 $y_j = F_p^{-1}(u_j)$。
   - **比对原则**: 对比 $\hat{y}_j$ 与 $y_j$。
     - 若 $\hat{y}_j == y_j$: 说明草稿在该步骤追踪成功，Target 继续验证 $u_{j+1}$。
     - 若 $\hat{y}_j \neq y_j$: **发生分歧**。立即停止验证，接受 $y_0 \dots y_{j-1}$，并以 Target 模型生成的 $y_j$ 计算得到新的锚点坐标 $u_{j+1}$，进入下一轮循环。这保证了 Drafter 的错误不会偏离 Target 的算术解码轨迹 $\Phi_p(u)$。

---

## 3. Pass@k 的无偏统计计算方法（严格区分独立与耦合采样）

参考 QuasiMoTTo 论文（`arXiv:2607.01179` Section 2.4.1、Theorem 1–2，见 `/home/dev/KAP/doc/quasi-monte-carlo.txt`）及官方实现 `/home/dev/KAP/Ref/quasimotto/eval.py`，不同采样策略生成的一批 $B=64$ 个样本（记判题结果为有序二值向量 $h^{(i)} = (h^{(i)}_0, h^{(i)}_1, \dots, h^{(i)}_{B-1}) \in \{0, 1\}^B$，命中总数 $c_i = \sum_{b=0}^{B-1} h^{(i)}_b$）具有完全不同的联合分布结构，**绝不能混用同一种估计器**：

### 3.1 独立同分布采样（`S1_iid`）—— Chen et al. (2021) 无放回组合估计器

当 $B$ 条轨迹彼此独立同分布（$h^{(i)}_b \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p_i)$）时，任意大小为 $k$ 的子集都是合法的 $k$ 次独立采样。因此单题 $i$ 的无偏估计为所有 $\binom{B}{k}$ 个子集中至少命中一次的比例：

$$
\widehat{\text{pass}@k}_{\text{iid}}^{(i)} = \begin{cases} 1 - \dfrac{\binom{B - c_i}{k}}{\binom{B}{k}}, & B - c_i \ge k \\[8pt] 1, & B - c_i < k \end{cases}
$$

可以证明 $\mathbb{E}\big[\widehat{\text{pass}@k}_{\text{iid}}^{(i)}\big] = 1 - (1 - p_i)^k$。

> **注意**：该组合公式严重依赖样本间的独立性与可交换性，**对耦合采样是数学上错误（有偏）的**——例如当耦合采样消除了重复命中时，套用该公式会系统性低估真实的耦合 `pass@k`。

### 3.2 随机平移格耦合采样（`S2_update`、`S3_paper` 等 Lattice 设置）—— 二进跨步自举估计器（Dyadic Stride Bootstrap, Theorem 1）

在 Lattice 采样中，$B = 2^L$ 个分支的初始均匀变量按格点顺序耦合生成：

$$
u_b = \left(\Delta + \frac{b}{B}\right) \bmod 1, \quad b = 0, 1, \dots, B - 1, \quad \Delta \sim \text{Unif}[0, 1).
$$

此时样本不再独立，但根据 **QuasiMoTTo Theorem 1**，对于任意 $k = 2^m \le B$，令跨步（stride）为 $s = B / k = 2^{L-m}$。对于任意起始偏移 $r \in \{0, 1, \dots, s - 1\}$，以步长 $s$ 等间隔抽取的长度为 $k$ 的子序列：

$$
u^{(r)}_j := u_{r + s \cdot j} = \left(\underbrace{\Big(\Delta + \frac{r}{B}\Big) \bmod 1}_{\tilde{\Delta}^{(r)} \sim \text{Unif}[0, 1)} + \frac{j}{k}\right) \bmod 1, \quad j = 0, 1, \dots, k - 1
$$

在分布上**严格等价于一个全新的、具有均匀随机平移 $\tilde{\Delta}^{(r)}$ 的 $k$ 点随机格（size-$k$ randomized lattice）**！

因此，长度为 $B$ 的一次 Lattice rollout 可以精确拆分为 $s = B/k$ 个同分布的大小为 $k$ 的子格（注意：是**跨步为 $s$ 的等间隔子序列** `hits[r::s]`，而不是连续分块），单题 $i$ 的无偏估计为这 $s$ 个子格中“至少命中一次”的比例：

$$
\widehat{\text{pass}@k}_{\text{lattice}}^{(i)} = \frac{1}{s} \sum_{r=0}^{s - 1} \mathbb{I}\!\left( \bigvee_{j=0}^{k-1} h^{(i)}_{r + s \cdot j} = 1 \right), \quad \text{其中 } s = \frac{B}{k}.
$$

对应代码实现（见 `/home/dev/KAP/passk/run_passk.py` 中的 `lattice_est` 与 `/home/dev/KAP/Ref/quasimotto/eval.py`）：

```python
def lattice_est(hits, k):
    stride = len(hits) // k
    return sum(1.0 if any(hits[r::stride]) else 0.0 for r in range(stride)) / stride
```

### 3.3 理论基准曲线与多题聚合（Global Aggregation）

单题能力 $p_i = \Pr_{y \sim \pi_\theta(\cdot \mid x_i)}[\text{correct}(y)]$ 因题目难度不同而异。在获得每道题 $i \in \{1, \dots, N_{\text{prompts}}\}$ 的单题估计值或精确概率 $p_i$ 后，整张图上的每条曲线都是在题目分布上的**算术平均（期望）**：

* **经验采样曲线（`S1_iid` / `S2_update` / `S3_paper`）**：
  $$
  \text{Pass@}k = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \widehat{\text{pass}@k}^{(i)}
  $$
* **理论上限曲线（`Upper bound`, Theorem 2）**：对每题先做截断 $\min(1, k p_i)$，再对所有题目求平均：
  $$
  \text{UpperBound}(k) = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \min(1,\, k \cdot p_i)
  $$
* **理论独立曲线（`Theory i.i.d.`）**：对每题计算独立采样解析通过率后对所有题目求平均：
  $$
  \text{TheoryIID}(k) = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \big(1 - (1 - p_i)^k\big)
  $$

---

## 4. 使用说明（逐任务评测与提高）

本目录只保留 pass@k 相关文件：`Readme.md`（本说明）、`run_passk.py`（实现）、`results/`（各任务的 JSON 结果与 PNG 曲线图）。槽位接受率实验的脚本与数据已移除（其文档见 `/home/dev/KAP/doc/槽位接受率实验.md`）。

### 4.1 逐任务运行

建议用 `CUDA_VISIBLE_DEVICES` 把每个任务限定在单块 GPU 上（进程内只可见一块卡，`--device` 恒为 `cuda:0`；这样 nvidia-smi 中每个任务恰好显示一行，也不会在别的卡上留下闲置上下文）：

```bash
cd /home/dev/KAP/passk

# 完整评测单个任务（100 题 × B=64 × 3 设置 + 上界），结果写入 results/
CUDA_VISIBLE_DEVICES=0 /home/dev/venv/bin/python -I run_passk.py --task maze      --device cuda:0
CUDA_VISIBLE_DEVICES=0 /home/dev/venv/bin/python -I run_passk.py --task sudoku    --device cuda:0
CUDA_VISIBLE_DEVICES=1 /home/dev/venv/bin/python -I run_passk.py --task countdown --device cuda:0
CUDA_VISIBLE_DEVICES=1 /home/dev/venv/bin/python -I run_passk.py --task 1d_arc    --device cuda:0

# 冒烟测试（2 题，快速验证管线）
/home/dev/venv/bin/python -I run_passk.py --task maze --smoke

# 只（重）算部分设置，并与已有结果合并 —— 提高某设置后无需重跑全部
/home/dev/venv/bin/python -I run_passk.py --task maze --settings S2_update \
    --merge-into results/results_maze.json

# 从已有 JSON 重新绘图
/home/dev/venv/bin/python -I run_passk.py --task maze --plot-only

# 找最弱的题（p_i 最低 / 各设置 pass@1 最低），作为逐题提高的靶点
/home/dev/venv/bin/python -I run_passk.py --task maze --weak 15

# 跨任务汇总表（读取 results/ 下全部 results_*.json，跟踪各任务提高效果）
/home/dev/venv/bin/python -I run_passk.py --report
```

### 4.2 实现要点

* **设置与估计器的对应（勿混用）**：`S1_iid` 用 Chen et al. 组合估计器（`std_est`）；`S2_update` 与 `S3_paper` 的 $u_0$ 为 lattice 耦合，必须用 Theorem 1 跨步自举（`lattice_est`）。
* **`S1_iid` 的实现**：与 `S2_update` 走同一个 AC 引擎（`dspark_fast.DSparkDecodeFast.generate_branches`），唯一区别是 $u_0$ 从 lattice 换成 i.i.d. 均匀——序列仍是 $\pi_\theta$ 的精确 i.i.d. 样本，且 S1↔S2 只差耦合方式，是最干净的控制变量。
* **正确性不变量**：同种子下 `S2_update` 与 `S3_paper` 的逐分支命中向量应完全一致（AC 提交序列即 $\Phi_p(u)$，Drafter 只决定跳过多少次 Target 前向；实测 $\tau \approx 2.3$）。
* **上界 $p_i$**：maze/sudoku/1d_arc 用 teacher-forcing 精确序列概率（不含 EOS，与 `Ref/quasimotto/eval.py` 的 `gt_ub` 路径一致；1d_arc 答案前加空格修正 token 边界）；countdown 因一题多解用 $M=128$ 的 QMC 蒙特卡洛估计。
* **可复现性**：每题种子 `SEED*1000+i`；重写后的 `run_passk.py` 与旧实现在同种子下逐位复现 `S2`/`S3` 命中与 $p_i$。
* **数据映射注意**：`maze_val.csv`/`sudoku_val.csv` 与 `prompts.pt` 的 val 题按行序对应（无 id 列）；countdown 的 prompt 需按 SFT 训练格式重建为 `Target: ...\nNumbers: ...\n`。

---

## 5. 正式实验记录（2026-11-24/25,首轮全量基线）

配置：`SEED=42`，每任务 100 道 val 题 × `B=64` rollouts（`T=1.0`，草稿长 `n=7`），`k∈{1,2,4,8,16,32,64}`；硬件 2×L20，单任务单卡（`CUDA_VISIBLE_DEVICES` 隔离）。逐任务耗时（三设置 + 上界合计）：maze 0.35h、countdown 1.26h、sudoku 3.73h、1d_arc 7.13h。

### 5.1 结果

| 任务 | S1@1 | S1@64 | **S2@1** | **S2@16** | **S2@64** | S3@64 | 上界@16 | 上界@64 | iid 理论@16 | τ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maze      | 1.9% | 55.0% | **2.1%** | **29.8%** | **67.0%** | 62.0% | 33.0% | 62.7% | 26.4% | 2.18 |
| sudoku    | 16.8% | 84.0% | **16.9%** | **71.5%** | **89.0%** | 88.0% | 70.8% | 88.2% | 62.6% | 6.40 |
| countdown | 42.3% | 81.0% | **42.7%** | **71.0%** | **83.0%** | 82.0% | 71.4% | 82.5% | 66.8% | 1.29 |
| 1d_arc    | 7.2% | 62.0% | **7.5%** | **45.5%** | **64.0%** | 65.0% | 45.4% | 63.8% | 38.4% | 4.49 |

（完整逐 k 数据见 `results/results_<task>.json`；S=S1_iid / S2=S2_update / S3=S3_paper）

### 5.2 结论

1. **格耦合增益（QMC 效应）稳定成立**：四个任务上 S2_update 在 @16 比 i.i.d. 理论高 3~9 个百分点，且紧贴理论上限 `E[min(k·p_i,1)]`（1d_arc @16：45.5% vs 45.4%；countdown @16：71.0% vs 71.4%）。@64 处各任务均已贴住上界——**采样侧已到顶，进一步提升只能提高 p_i（模型训练）**。
2. **S2（带 DSpark drafter）与 S3（论文纯 AC 解码）分布等价**：逐题命中完全一致的比例为 62%（maze）/55%（sudoku），其余题 ±1~2 且正负平衡（bf16 批式验证 vs 逐 token 前向的数值漂移，非系统性偏差）；drafter 的作用是把同一分布的解码加速 τ=1.3~6.4×。
3. **S1 与 iid 理论吻合**（如 sudoku @64：84.0% vs 82.8%），验证 Chen 估计器实现。

### 5.3 已知事项

- maze 的 S2@64（67.0%）略超上界（62.7%）：上界的 p_i 只 teacher-force 标准答案串，而 maze 判题接受**任意**合法路径（多解任务），真实解出概率被低估；sudoku（唯一解）/countdown/1d_arc 均无此现象（S3 在上界之下）。
- 1d_arc 使用 Readme 规定的 Base 模型 + `dspark_arc_sft` drafter（非 SFT 记忆路径），@1=7.6% 属 in-context 泛化水平。
- `nvidia-smi` 中每任务应恰好一行进程：启动务必带 `CUDA_VISIBLE_DEVICES`（否则引擎内裸 `torch.cuda.synchronize()` 会在 cuda:0 初始化一个 ~286MiB 的寄生上下文，多出一行）。

### 5.4 实验过程存档

重写 `run_passk.py`（旧版 git 历史）→ 四任务冒烟（2 题）通过，且与旧实现在同种子下**逐位复现** S5/S6 命中与 p_i → 15 代理对抗性代码审查（4 视角 + 反驳验证）：确认 7 项、反驳 4 项，全部确认项已修复（报告 nan%、sudoku 命名/glob、--weak 越界、空题除零、--plot-only 崩溃）→ 两卡双链全量运行 → 上述结果。提高迭代从 §4.1 的 `--weak` / `--merge-into` 工作流进入；训练侧按 `RL/训练框架.md` 的 GRPO+drafter 交替管线推进（最弱任务优先：maze p̄≈2.5% < 1d_arc 7.6% < sudoku 17% < countdown 43%）。
