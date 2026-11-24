# DSpark 多分支推理引擎：分层逆 CDF 采样 + 批量验证 + 数学回滚

> 实现文件：`/home/dev/KAP/src4dspark/dspark_fast.py`（入口 `DSparkDecodeFast.generate(..., branches=B, mode='stratified')`）
> 校验脚本：`/home/dev/KAP/test/test_dspark_fast.py`（4 项测试全过）；基准：`/home/dev/KAP/test/dspark_fast_speedup.ipynb`
> 文档日期：2026-11-16；2026-11-20 采样层改为逆 CDF（见 §3.7）；2026-11-21 起引擎内 Gumbel 代码路径已全部删除，对照基线为 `mode='iid'`（独立均匀数 + 同一逆 CDF 链）
> 结果速览（12 prompt × 256 token，T=1.0，baseline 57.2 tok/s）：B=4 → 93.2 tok/s（1.63×）；B=8 分层采样 → **105.6 tok/s（1.85×）**；代码类 prompt 最高 113.9 tok/s（2.0×）。全程无损。

---

## 1. 记号总表（每个符号的含义、形状、出处）


| 符号                              | 含义                                                                                                                    | 形状 / 取值             | 出处                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------ |
| `x₀`（anchor / bonus）           | 上一轮 target 自己采出、本轮必然提交的 token；验证块的第一个输入                                                        | 标量 token id           | 上一轮 verify 的末位 logits 或拒绝轮的修正 token |
| `x_j`，j=1..n                     | 第 j 个草稿 token                                                                                                       | 标量 token id           | drafter 采样                                     |
| `n`（γ）                         | 块长 = 草稿槽位数                                                                                                       | 7（`ND`）               | `dspark_paper.ND`                                |
| `B`                               | 分支数（每轮独立采样的草稿链条数）                                                                                      | 1,2,4,8…               | `generate(branches=B)`                           |
| `U_j`                             | drafter 并行 backbone 在槽 j 的 base logits。**所有分支共享**（backbone 只跑一次）                                      | (V,)，V=248,320         | `d.backbone(...)` → `d.lm_head(z)`              |
| `markov(v)`                       | bigram 转移偏置：`W1[v] @ W2ᵀ`。把"上一 token 是 v"的串行依赖注入当前槽                                                | (V,)，rank r=128        | `d.markov(prev)`，`dflash_big.MarkovHead`        |
| `q_j(·|x_{j-1})`                 | 槽 j 的提案分布 =`softmax((U_j + markov(x_{j-1}))/T)`。**base 共享、markov 按分支查表**——这是分支间唯一的分布差异来源 | (B,V) 批量              | `_propose_multi`                                 |
| `T`                               | 温度。采样、提案分布、接受判定三方同温                                                                                  | 1.0（论文口径）         | 引擎构造参数                                     |
| `block`（块/路径/链）             | 一条分支的 7-token 序列`(x₁..x₇)`                                                                                     | V⁷ 空间中的一个点      | —                                               |
| `P(block)`                        | 块概率 =`Π_j q_j(x_j|x_{j-1})`（一阶链连乘）                                                                           | ∈(0,1)                 | `st['draft_lp'] = log P`                         |
| `F`                               | 块分布按**字典序**铺在 [0,1) 上的 CDF；**每个块的区间宽度 = 它的概率**                                                  | [0,1) 划分              | §2                                              |
| `u_b`                             | 分支 b 的耦合随机数：`u_b = (π(b)+v_b)/B`，π 为每轮新抽的 B 层随机置换，`v_b ~ U(0,1)` 独立。**置换必需**：没有它分支 b 被钉死在第 b 层，边际不再是 U(0,1)，提案分布偏离 q | ∈[0,1)                | `_propose_multi(mode='stratified')`              |
| `u₀`                              | anchor 采样用的均匀数：`_sample` 内 `torch.rand` 新抽，逆 CDF 选 token 后**不消费残差**（每个 anchor 独立入口）                       | ∈(0,1)               | `_sample`（2026-11-20 改为逆 CDF，原为 Gumbel-max） |
| `u ← (u − F⁻(x))/q(x)`         | 嵌套残差投影：走进所选 token 的子区间后重新归一化到 [0,1)                                                               | 标量                    | 链式逆 CDF 的每步（首槽 u 来自 u₀ 或 u_b，后续都是残差） |
| `p_j`                             | **target** 在位置 pos+j 的真实分布（T 温度）。来自批量 verify 前向的第 j−1 行 logits                                   | (B,n,V)                 | `_forward` 返回的 `v_lg`                         |
| `α_j = Σ_c min(p_j(c), q_j(c))` | 槽 j 的接受概率（teacher-forced 口径；`1 − TV/2`）                                                                     | ∈[0,1]                 | 评测/τ 的原料                                   |
| `a = min(1, p(x)/q(x))`           | 已提案 token x 的接受比率（clamp 到 1）                                                                                 | (B,n)                   | `_accept_multi`                                  |
| `corr`（修正 token）              | 拒绝槽上从残差分布`(p−q)⁺/Z` 采出的 token；它同时是下一轮的 anchor                                                    | 标量                    | `_accept_multi`                                  |
| `Z = Σ_c (p(c)−q(c))⁺`         | 残差质量（= TV 距离；拒绝可能发生时必有 Z>0）                                                                           | 标量                    | 同上                                             |
| `τ`                              | 每轮提交的 token 数（含 anchor 与修正）= 加速的直接乘数                                                                 | 实测 2.4→4.4           | `st['tokens']/st['rounds']`                      |
| `S_t`                             | 一个 GDN 层在吃进第 t 个 token 后的循环状态 (16,128,128) fp32                                                           | 每层一份                | cache.recurrent_states                           |
| `snap`                            | 轮开始前 18 层 GDN 的 (conv, recurrent) 克隆                                                                            | 18×2 个小张量 ≈19.8MB | `GdnRewinder.snap`                               |

---

## 2. 块分布与它在 [0,1) 上的表示

drafter 生成块的概率按链式法则分解：

```
P(x₁..x₇ | x₀) = q₁(x₁|x₀) · q₂(x₂|x₁) · … · q₇(x₇|x₆)
```

把所有块按**字典序**排好铺在 [0,1) 上：以 token i 开头的所有块恰好占据宽度为 `q₁(i)` 的连续区间，区间内部再按 `q₂(·|i)` 细分，依此嵌套。得到的核心恒等式：

> **每个块的区间宽度 = 它的精确概率 P(block)。**

玩具例子（2 slot，20 万次采样验证过边际精确，见 quickstart §5.5）：

```
q₁ = [0.6, 0.3, 0.1]（token a/b/c）
q₂(·|a)=[0.5,0.5]  q₂(·|b)=[0.9,0.1]  q₂(·|c)=[0.2,0.8]

块区间：(a,d)[0.00,0.30) (a,e)[0.30,0.60) (b,d)[0.60,0.87)
        (b,e)[0.87,0.90) (c,d)[0.90,0.92) (c,e)[0.92,1.00)
```

V⁷ 的块空间无法枚举，但**不需要枚举**：一阶链结构让"查某个块的区间"可以用 7 步 cumsum/searchsorted 完成。

---

## 3. 单分支：一个 uniform 走出整条路径（嵌套逆 CDF）

### 3.1 问题设定：块分布是一阶链

drafter 对一个块 $(x_1,\dots,x_n)$（$n=7$，$x_0$ 为 anchor）的概率按链式法则分解：

$$
P(x_{1:n}\mid x_0)=\prod_{j=1}^{n}q_j\!\left(x_j\mid x_{j-1}\right),
\qquad
q_j(\cdot\mid x_{j-1})=\mathrm{softmax}\!\left(\frac{U_j+\mathrm{markov}(x_{j-1})}{T}\right)\in\Delta^{V-1}
$$

其中 $U_j$ 是并行 backbone 的槽位 base logits（**全分支共享**），$\mathrm{markov}(x_{j-1})=W_1[x_{j-1}]W_2^{\top}\in\mathbb{R}^V$ 是按上一 token 查表的 bigram 偏置。目标：**只用一个** $u\sim\mathcal{U}(0,1)$，得到一条以 $P(x_{1:n}\mid x_0)$ 为分布的路径。

### 3.2 块的嵌套区间（递归定义）

对前缀 $x_{1:k}$，记它的区间为 $\mathcal{I}(x_{1:k})=[L_k,\;L_k+w_k)$，由递归给出：

$$
L_0=0,\;\; w_0=1;\qquad
L_k = L_{k-1} + w_{k-1}\,F_j\!\left(x_j\mid x_{j-1}\right),\qquad
w_k = w_{k-1}\,q_k\!\left(x_k\mid x_{k-1}\right)
$$

其中 $F_j(c\mid x_{j-1})=\sum_{c'<c}q_j(c'\mid x_{j-1})$ 是槽 $j$ 在 $c$ 之前的累积质量（前缀和）。**归纳可证**：同一层所有 $c$ 的子区间恰好不重叠地铺满父区间（因为 $\sum_c q_j(c\mid x_{j-1})=1$），所以走到第 $n$ 层时

$$
w_n=\prod_{j=1}^{n}q_j(x_j\mid x_{j-1})=P(x_{1:n}\mid x_0)
$$

——**每个块的区间宽度 = 它的精确概率**，全部叶子无重叠地覆盖 $[0,1)$。

### 3.3 逆走链：算法

给定 $u\sim\mathcal{U}(0,1)$，逐步做三件事（`_propose_multi(mode='stratified')` 的内层，批量 $\times B$）：

**① 逆 CDF 选 token**：

$$
x_j=\mathrm{CDF}_j^{-1}(u)=\min\Big\{c:\ \mathrm{cdf}_j(c)>u\Big\},
\qquad \mathrm{cdf}_j(c)=F_j(c\mid x_{j-1})+q_j(c\mid x_{j-1})
$$

（代码：`x = searchsorted(cumsum(q), u, right=True)`，即 $u$ 落在谁的区间就选谁。）

**② 残差投影**：把 $u$ 换算成它在 $x_j$ 子区间里的相对位置，重新归一化回 $[0,1)$：

$$
u \;\leftarrow\; \frac{u-F_j(x_j\mid x_{j-1})}{q_j(x_j\mid x_{j-1})}
$$

**③ 用更新后的 $u$ 进入下一槽**（槽 $j{+}1$ 的 CDF 以 $x_j$ 为条件）。

### 3.4 关键引理：残差仍是均匀分布

> **引理**：设 $U\sim\mathcal{U}(0,1)$，$[A,A+w)\subset[0,1)$。则
> $$\Pr\!\left[\frac{U-A}{w}\le t \;\middle|\; U\in[A,A+w)\right]=t,\quad t\in[0,1]$$
> 即**条件在子区间上再归一化的 uniform 仍是 uniform**。

证明一行：$\Pr[U\le A+tw\mid U\in[A,A+w)] = \frac{tw}{w}=t$。

这条引理保证递归自相似：每一步"消费"掉 $u$ 中属于已选 token 的质量后，剩下的部分对该槽条件分布而言是一个全新均匀样本。于是走链映射满足

$$
\Pr\big[\mathrm{walk}(u)=x_{1:n}\big]=w_n=P(x_{1:n}\mid x_0)
$$

——**路径是块分布的精确采样**（这就是无损性的根基：每槽提案边际恰为 $q_j$，后续 $\min(1,p_j/q_j)$ 的接受判定照常成立）。

### 3.5 手算例子（玩具分布，$u=0.35$）

$q_1=[0.6,0.3,0.1]$（a/b/c），$q_2(\cdot|a)=[0.5,0.5]$（d/e）：

$$
\mathrm{cdf}_1=[0.6,\,0.9,\,1.0]\ \Rightarrow\ x_1=a;\qquad
u'=\frac{0.35-0}{0.6}=0.58\overline{3}
$$
$$
\mathrm{cdf}_2(\cdot|a)=[0.5,\,1.0]\ \Rightarrow\ x_2=e;\qquad \text{块 }(a,e)\in[0.30,0.60),\ 0.35\ \text{的相对位置正是}\ 0.58\overline{3}
$$

### 3.6 数值实现与 Gumbel 的关系

- 代码只做三件事：`cdf = q.cumsum(-1)`；`x = searchsorted(cdf, u, right=True).clamp(max=V−1)`；`u = ((u − lo)/qx).clamp(0, 1−1e−7)`。每槽 $O(V)$ cumsum + $O(\log V)$ 查找，按分支批量。
- 对**单次**选择，`argmax(log q + G)`（$G=-\log(-\log u)$ 为 Gumbel 逆 CDF）与逆 CDF 同分布；但**嵌套残差只有逆 CDF 形式能做**（Gumbel 的噪声不落在块概率的区间结构上），所以路径采样用逆 CDF。
- `markov(x_{j-1})` 的变化范围：base logits 全分支共享；偏置按各自上一 token 查表。槽 1 时所有分支的 $x_0$ 相同（分布一致，差异全在 $u$）；两分支在槽 $j$ 采出不同 token 后，槽 $j{+}1$ 起分布不同——"分支在 markov 采样处分化"。

### 3.7 完整采样流水线：从 anchor 到末槽（2026-11-20 更新）

**原实现（已删除）**：anchor token 曾用 Gumbel-max 采样（`_sample`：`argmax(row/T + G)`，$G=-\log(-\log u)$）；draft slots 曾有一个 `mode='gumbel'` 分支，每槽、每 token 重新抽独立 Gumbel 噪声。两者都无法传播残差，且已于 2026-11-21 从代码中彻底删除。

**当前实现**：整个采样层统一使用逆 CDF，残差在 anchor 与各槽之间串行传播：

```
① anchor 采样（_sample，每轮轮开始或拒绝后）：
   u₀ ~ U(0,1)        ← 本轮唯一的随机数入口
   p  = softmax(row / T)
   F  = p.cumsum(-1)
   x₀ = searchsorted(F, u₀, right=True)

② 进入 _propose_multi/single（mode='stratified'）：
   u = u₀（单分支）  或  u_b = (π(b) + v_b)/B（多分支；π 每轮随机置换 B 个层，v_b ~ U(0,1) 独立）
   for j = 1..n:
       q_j = softmax((U_j + markov(x_{j-1})) / T)
       F_j = q_j.cumsum(-1)
       x_j = searchsorted(F_j, u, right=True)
       lo  = F_j[x_j] - q_j[x_j]          # x_j 区间左边界
       u   ← (u - lo) / q_j[x_j]          # 残差：映射回 [0,1)
```

关键性质：
- **第一个 u 是随机的**（anchor 轮由 `torch.rand` 新抽；多分支由分层均匀 `v_b` 产生）。
- **后续的 u 是残差**——每消费一个 token 的概率质量后，剩余部分重新归一化，仍是 $U(0,1)$（§3.4 引理）。
- 引擎内**已无任何 Gumbel 路径**（2026-11-21 起 `_propose_multi` 删除全部 Gumbel 代码）。对照基线改用 `mode='iid'`：每条分支独立抽 `u_b ~ U(0,1)`，走同一条逆 CDF 链——与 `mode='stratified'` 的唯一区别是随机数的抽取方式（独立 vs 分层），两者都是逆 CDF。
- 修正 token（`corr`）由拒绝槽的残差 CDF 决定，不受此处改动影响（逻辑在 `_accept_multi`，同样用 `searchsorted`）。

---

## 4. 多分支：分层 uniform（耦合随机数）—— 按概率比例覆盖块分布

B 条分支的随机数**不是独立抽取**，而是分层：

```
u_b = (b − 1 + v_b) / B,   v_b ~ U(0,1) 独立,  b = 1..B
```

效果：[0,1)（=块概率质量）被切成 B 个互不重叠的层，每条分支负责一层。

- **覆盖性质**：宽度 ≥ 1/B 的块**必然**被某条分支采到；小概率块按比例覆盖。i.i.d. 采样（mode='iid'，独立均匀数 + 同一逆 CDF 链）会重复抽中同一个高概率块，有效覆盖 < B。
- **无损性保持的严格理由**：`v_b` 独立均匀 ⇒ `u_b` **边际均匀** ⇒ 每条路径的边际分布仍是 P(block) ⇒ 每分支的接受判定依旧精确。**耦合只改变分支之间的相关性（覆盖），不改变任何一条分支自己的分布。**
- ⚠️ 不可用的替代：强制首 token 互异的无放回采样（Gumbel top-k）会让分支的提案边际 ≠ q，接受判定有偏。
- 分支的 `P(draft)` 在走链时顺带累加（`st['draft_lp']` = log Π q_j），想要"不同块的概率"时直接读。

---

## 5. 批量验证 + 单 uniform 融合的接受/修正

B 条链作为 **batch=B 的 B 个序列**送进**同一个** target 前向（每分支 8 个 token：`[x₀, x₁..x₇]`）：

```
verify:  (B, 8) tokens, position_ids 相同, GDN/KV 状态按 batch 复制 B 份
p_j      = softmax(v_lg[b, j−1] / T)     # 槽 j 的 target 分布 = 第 j−1 行 logits
```

**每槽一个 uniform 同时决定"接不接受"和"拒绝修正成什么"**。给定走到槽 j，结果分布恰好划分 [0,1)：

```
接受 x_j          :  a = min(1, p(x_j)/q(x_j))
拒绝且修正为 c    :  (1 − a) · (p(c) − q(c))⁺ / Z
```

一个 uniform u：`u < a` → 接受；否则用 `(u − a)/(1 − a)` 在 `(p−q)⁺` 的 CDF 上 searchsorted 直接得到修正 token（拒绝可能发生 ⇔ `(q(x)−p(x))⁺ > 0` ⇔ Z>0，除法安全）。每槽 2 个随机操作合并成 1 个。

**提交规则**：取**接受最深**的分支（`_accept_multi` 对 (B,n) 的首拒绝位置取 argmax），提交它的前缀 + 它的修正 token；修正 token 同时成为下一轮的 anchor（它是 target 分布的合法采样）。每轮**恰好 1 次** target 前向。

**无损性**：每条分支的接受判定是标准的逐槽拒绝采样（提案边际 = q），多分支只是多个独立合法赌注；取最深者不改变任何单分支的输出分布。

---

## 6. GDN 状态的数学回滚（拒绝时不需要第二次前向）

Qwen3.5 的 18 层 Gated DeltaNet 维护**循环状态**（整条前缀的压缩摘要，无法像 attention KV 那样按 token 裁剪）。拒绝发生时：

1. **快照**：轮开始前克隆 18 层的 (conv, recurrent) 状态（~0.2ms；attention KV 是逐 token 追加的，直接 crop）。
2. **数学重建**：gated delta rule 的更新对前缀满足**结合律**——
   `S(接受前缀) = RecurStep(S_快照, 已接受 token 的因子)`。
   因子（post-conv 的 q/k/v、β、g）从 **verify 前向已经收集的 hidden_states 在层外重算**（每层 5 个小 GEMM，18 层堆成一次 batched bmm），然后一次 `fused_recurrent_gated_delta_rule(initial_state=快照)` 批量调用得到接受前缀末的状态。与真实前向的最大误差 2.6e-3（bf16 噪声级别；HF 自己 prefill 用 chunked、decode 用 recurrent，同样存在）。
   实测 ~1.1ms，替代老引擎 ~28.6ms 的重放前向。
3. **conv 状态**：`conv_state(新) = cat(快照conv, 已接受 token 的 mixed_qkv)[..., -4:]`（注意窗口 off-by-one：新位置 t 的窗口是 `full_conv[t+1:t+5]`，causal_conv1d_fn 的权重方向是 w[K−1] 对应当前位置）。
4. **为什么不能真用树 attention**：tree attention 只能隔离 6 个全注意力层；18 个 GDN 层的状态按序列顺序累积，mask 隔离不了分支。所以分支在 **batch 维隔离**（各自一份状态副本），等价于"每条分支一条独立链"，而不是共享状态的树。

---

## 7. 每轮成本分解（实测，L20，单请求）


| 环节                               | B=1         | B=8            | 说明                                                                                    |
| ---------------------------------- | ----------- | -------------- | --------------------------------------------------------------------------------------- |
| drafter base logits + markov 采样  | 6.0–6.6 ms | 6.8–8.9 ms    | backbone 一次；采样每槽 (B,V) 批量；stratified 的 cumsum/searchsorted 比 gumbel 贵 ~2ms |
| **target 批量 verify 前向**        | **~30 ms**  | **~30–31 ms** | **B=1→8 墙钟不变**（CPU 发射瓶颈 1798 kernel，GPU 仅忙 ~6ms）——多分支验证近免费      |
| 状态复制（GDN repeat + KV repeat） | 0           | ~2–3 ms       | batch 维 B 份                                                                           |
| 数学回滚（仅拒绝轮）               | 1.1 ms      | 0.8–1.1 ms    | 替代老引擎 28.6ms 重放                                                                  |
| 每轮合计                           | ~37.7 ms    | ~40 ms         | τ 2.43 → 4.11–4.44                                                                   |

盈亏平衡 τ = 每轮毫秒 / baseline 每 token 毫秒 ≈ 2.1；B=8 时 τ 4.44 ≫ 2.1。

---

## 8. 实测汇总

12 prompt × 256 token，T=1.0，v2-B drafter（baseline 57.2 tok/s）：


| 配置                   | τ_real  | tok/s     | vs baseline          |
| ---------------------- | -------- | --------- | -------------------- |
| 老引擎（2 前向/轮）    | 3.79     | 56.5      | 0.99×               |
| 快速引擎 B=1（Gumbel） | 2.43     | 62.5      | 1.09×               |
| B=4（Gumbel）          | 3.64     | 93.2      | 1.63×               |
| B=4（分层逆 CDF）      | 3.72     | 87.0      | 1.52×（采样贵 2ms） |
| **B=8（分层逆 CDF）**  | **4.44** | **105.6** | **1.85×**           |

- 域分化：代码类 prompt B=4 达 113.9 tok/s（2.0×）； drafter 训练分布外的文本（数字网格）接受率≈0，反而慢——扩域训练数据是正交杠杆。
- 无损性三重验证：每轮 verify logits 与"从零重算"金标准偏差与老引擎同级（mean 0.29 vs 0.30）；回滚状态与真实前向差 2.6e-3；任务级正确率与 HF generate 一致（quickstart §5.3）。

---

## 9. 符号 → 代码对照


| 文档符号                | 代码位置（`dspark_fast.py`）                                                        |
| ----------------------- | ----------------------------------------------------------------------------------- |
| U_j（base logits）      | `_propose_multi` 里 `base = d.lm_head(z...)`                                        |
| markov(x_{j-1})         | `self.d.markov(prev)`（(B,V) 批量查表）                                             |
| u₀ / anchor 采样        | `_sample`：`u=rand(); p=softmax(row/T); searchsorted(p.cumsum(-1), u)`（**原为 Gumbel-max，2026-11-20 改**） |
| u_b / 残差投影          | `mode='stratified'` 分支：`u = (arange(B)+v)/B`，每步 `searchsorted` + `(u−lo)/qx` |
| 验证 chunk              | `chunk = torch.cat([anchor.view(1,1).expand(B,1), dr], 1)`                          |
| p_j / v_lg              | `_forward` 返回的 `v_lg (B,n+1,V)`                                                  |
| a / u / 修正            | `_accept_multi`：`a=(pt/qt).clamp(max=1)`，`u<a` 判定，残差 CDF `searchsorted`      |
| 最深分支                | `first.argmax()` → `bidx`；`_select_branch(cache, bidx)`                           |
| snap / restore / rewind | `GdnRewinder.snap/restore/rewind(…, bidx)`（rewind 的 hidden_states 取该分支的行） |
| 状态批量复制            | `_expand_cache(cache, B)`（attention repeat + GDN repeat）                          |

已知未做：target 前向的 CUDA graph/静态 KV（发射开销再砍 ~20ms，预计总体 3×+）；drafter 扩域训练（数字/推理文本）；置信度截断（论文 prefix scheduler）与多分支正交，可叠加。

### 8.1 分层采样 vs i.i.d. 的接受率专项测试（`test/test_stratified_accept.py`）

**边际精确性**（Part 1，决定"提案分布是否无偏"）：

| 检验 | stratified | 独立 uniform（=Gumbel 的等价形式） |
| --- | --- | --- |
| 槽 1 频率 vs 解析 q₁（M=20 万） | TV **0.0394** | TV 0.0401 |
| 槽 2 条件频率 E[x₂\|x₁=c*]（M=10 万） | TV **0.0021** | TV 0.0044 |

两者同为采样噪声级别（stratified 甚至略更近解析值）——耦合没有引入任何边际偏差。

**端到端接受率**（Part 2，B=4，12 prompt × 256 token，T=1.0）：

| 模式 | mean accepted/round | τ | P(k≥1..7) |
| --- | --- | --- | --- |
| gumbel（i.i.d.） | 2.472 | 3.669 | 0.85 0.63 0.43 0.27 0.16 0.08 0.05 |
| stratified | **2.563** | **3.841** | 0.84 0.63 0.44 **0.30 0.18 0.11 0.06** |

浅槽持平、深槽略升（P(k≥5/6/7) 0.16/0.08/0.05 → 0.18/0.11/0.06）：按概率比例覆盖让高质量分支更多存活到深层。**结论：接受率不降反微升，τ +4.6%。**

### 8.2 域扩训练后的跨域实测（2026-11-16）

训练：PB 49,376 + countdown 20,000 + ARC 20,000 混训（v2-B 权重继续，lr 3e-4 cosine，2 epochs = 9,080 步，3.1h；`parameter/draft/dspark_domain/best.pt`）。

**T=1.0，branches=8 分层采样**（`eval_before/after.json`）：

| 域 | v2-B | domain | 变化 |
| --- | --- | --- | --- |
| countdown | 93.8 tok/s，τ 4.00 | **130.7 tok/s，τ 5.61** | +39% |
| 1d_arc | 32.0 tok/s，τ 1.78 | **39.6 tok/s，τ 2.02** | +24% |
| PB | 89.1 tok/s，τ 3.73 | 89.5 tok/s，τ 3.78 | 持平（混训防遗忘 ✓） |

**T=0.7，加速比与正确率**（`speed_acc.json`，20 题每域，HF 逐 token 为 baseline）：

| 域 | HF | DSpark[v2-B] | DSpark[domain] |
| --- | --- | --- | --- |
| countdown | 58.3 tok/s，0/20 | 103.1 tok/s，**1.77×**，0/20 | 147.1 tok/s，**2.52×**，1/20 |
| 1d_arc | 57.0 tok/s，1/20 | 28.7 tok/s，**0.50×**，0/20 | 39.0 tok/s，**0.68×**，0/20 |

正确率各 config 完全一致（无损性的任务级第四次确认；绝对值 0–5% 是 Base 模型本身能力）。countdown 加速比 1.77→2.52×；ARC 仍低于盈亏平衡（τ 2.02 < ~2.3，基模型的 `<think>` 长推理文本依旧难匹配），需要继续加域内数据或降低该域草案长度。
