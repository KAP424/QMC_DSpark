# Pass@k 评测与绘图说明 (2026-11-24)

本目录用于对 **Maze**、**Sudoku**、**Countdown** 和 **1D-ARC** 四个推理任务进行投机采样（Speculative Decoding）与算术编码准蒙特卡洛（QuasiMoTTo / DSpark AC）的 `pass@k` 性能评测与可视化。

---

## 1. 评测模型与数据集配置

所有任务均采用默认抽样 `N_prompts = 100` 道验证集题目，每题并行生成 `B = 64` 条轨迹（`T = 1.0`，草稿长度 `n = 7`），计算 $k \in \{1, 2, 4, 8, 16, 32, 64\}$ 的 `pass@k`：

| 任务 | Target 模型路径 | Drafter 参数路径 | 评测验证集数据源 | 单题真实通过率 $p_i$ 计算方式 |
| :--- | :--- | :--- | :--- | :--- |
| **Maze** | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-maze-sft/hf_model` | `/home/dev/KAP/parameter/draft/dspark_maze_sft/best.pt` | `/home/dev/KAP/data/maze_sudoku/prompts.pt` (`split='val'`) + `maze_val.csv` | Teacher-forcing 精确序列概率 $\pi_\theta(\text{sol}\mid\text{prompt})$ |
| **Sudoku** | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model` | `/home/dev/KAP/parameter/draft/dspark_sudoku_sft/best.pt` | `/home/dev/KAP/data/maze_sudoku/prompts.pt` (`split='val'`) + `sudoku_val.csv` | Teacher-forcing 精确序列概率 $\pi_\theta(\text{sol}\mid\text{prompt})$ |
| **Countdown** | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model` | `/home/dev/KAP/parameter/draft/dspark_countdown_sft/best.pt` | `/home/dev/KAP/data/domain_exp/prompts.pt` (`split='val'`) | 高精度 QMC 蒙特卡洛估计（`M = 128` rollouts，因一题存在多个合法算式） |
| **1D-ARC** | `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original` | `/home/dev/KAP/parameter/draft/dspark_arc_sft/best.pt` | `/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset/*/*.json` | Teacher-forcing 精确序列概率 $\pi_\theta(\text{ans}\mid\text{prompt})$ |

> **注**：`/home/dev/KAP/data/passk_filtered/` 仅用于过滤 `train` 集做后续 GRPO 强化学习训练，**不用于**此处 `pass@k` 测试集评估。

---

## 2. 采样设置定义与 S5_update 的精确流程机制

绘图包含以下 5 条对比曲线：
1. **`S5_update`**：DSpark 算术编码确定性解码（AC）。
2. **`S6_paper`**：无 Drafter 的原论文基准。
3. **`S1_iid`**：传统独立同分布投机采样。
4. **`Upper bound`**：理论上线 $\mathbb{E}_{i}[\min(k \cdot p_i, 1)]$。
5. **`Theory i.i.d.`**：理论独立采样 $\mathbb{E}_{i}[1 - (1 - p_i)^k]$。

### S5_update (AC) 核心流程：
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
$$\widehat{\text{pass}@k}_{\text{iid}}^{(i)} = \begin{cases} 1 - \dfrac{\binom{B - c_i}{k}}{\binom{B}{k}}, & B - c_i \ge k \\[8pt] 1, & B - c_i < k \end{cases}$$
可以证明 $\mathbb{E}\big[\widehat{\text{pass}@k}_{\text{iid}}^{(i)}\big] = 1 - (1 - p_i)^k$。
> **注意**：该组合公式严重依赖样本间的独立性与可交换性，**对耦合采样是数学上错误（有偏）的**——例如当耦合采样消除了重复命中时，套用该公式会系统性低估真实的耦合 `pass@k`。

### 3.2 随机平移格耦合采样（`S5_update`、`S6_paper` 等 Lattice 设置）—— 二进跨步自举估计器（Dyadic Stride Bootstrap, Theorem 1）
在 Lattice 采样中，$B = 2^L$ 个分支的初始均匀变量按格点顺序耦合生成：
$$u_b = \left(\Delta + \frac{b}{B}\right) \bmod 1, \quad b = 0, 1, \dots, B - 1, \quad \Delta \sim \text{Unif}[0, 1).$$
此时样本不再独立，但根据 **QuasiMoTTo Theorem 1**，对于任意 $k = 2^m \le B$，令跨步（stride）为 $s = B / k = 2^{L-m}$。对于任意起始偏移 $r \in \{0, 1, \dots, s - 1\}$，以步长 $s$ 等间隔抽取的长度为 $k$ 的子序列：
$$u^{(r)}_j := u_{r + s \cdot j} = \left(\underbrace{\Big(\Delta + \frac{r}{B}\Big) \bmod 1}_{\tilde{\Delta}^{(r)} \sim \text{Unif}[0, 1)} + \frac{j}{k}\right) \bmod 1, \quad j = 0, 1, \dots, k - 1$$
在分布上**严格等价于一个全新的、具有均匀随机平移 $\tilde{\Delta}^{(r)}$ 的 $k$ 点随机格（size-$k$ randomized lattice）**！

因此，长度为 $B$ 的一次 Lattice rollout 可以精确拆分为 $s = B/k$ 个同分布的大小为 $k$ 的子格（注意：是**跨步为 $s$ 的等间隔子序列** `hits[r::s]`，而不是连续分块），单题 $i$ 的无偏估计为这 $s$ 个子格中“至少命中一次”的比例：
$$\widehat{\text{pass}@k}_{\text{lattice}}^{(i)} = \frac{1}{s} \sum_{r=0}^{s - 1} \mathbb{I}\!\left( \bigvee_{j=0}^{k-1} h^{(i)}_{r + s \cdot j} = 1 \right), \quad \text{其中 } s = \frac{B}{k}.$$
对应代码实现（见 `/home/dev/KAP/passk/sample/run_settings.py` 与 `/home/dev/KAP/Ref/quasimotto/eval.py`）：
```python
def lattice_est(hits, k):
    stride = len(hits) // k
    return sum(1.0 if any(hits[r::stride]) else 0.0 for r in range(stride)) / stride
```

### 3.3 理论基准曲线与多题聚合（Global Aggregation）
单题能力 $p_i = \Pr_{y \sim \pi_\theta(\cdot \mid x_i)}[\text{correct}(y)]$ 因题目难度不同而异。在获得每道题 $i \in \{1, \dots, N_{\text{prompts}}\}$ 的单题估计值或精确概率 $p_i$ 后，整张图上的每条曲线都是在题目分布上的**算术平均（期望）**：
* **经验采样曲线（`S1_iid` / `S5_update` / `S6_paper`）**：
  $$\text{Pass@}k = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \widehat{\text{pass}@k}^{(i)}$$
* **理论上限曲线（`Upper bound`, Theorem 2）**：对每题先做截断 $\min(1, k p_i)$，再对所有题目求平均：
  $$\text{UpperBound}(k) = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \min(1,\, k \cdot p_i)$$
* **理论独立曲线（`Theory i.i.d.`）**：对每题计算独立采样解析通过率后对所有题目求平均：
  $$\text{TheoryIID}(k) = \frac{1}{N_{\text{prompts}}} \sum_{i=1}^{N_{\text{prompts}}} \big(1 - (1 - p_i)^k\big)$$
EOF
