# DSpark 投机解码 · 架构与参数说明

> 目标模型：`Qwen3.5-0.8B-Base`（原始参数的只读拷贝：`/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original`；下载位置 `/home/dev/models/Qwen3.5-0.8B-Base`）
> 参考：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://hustzxd.github.io/EfficientPaper/notes/2026/DSpark/note/)
> 代码（旧版，4 层、4 位置窗口）：`/home/dev/KAP/src4dspark/dflash_big.py`，权重：`/home/dev/KAP/parameter/draft/legacy/dflash_big.pt`
> 对照论文重写的新版：`src4dspark/dspark_paper.py` + `train/dspark_loss.py`（5 层、整条前缀上下文，权重 `parameter/draft/dspark_pb/dflash_pb_best.pt`；训练流程和文件布局见 `DSpark-训练文档.md`），见 §9
> 文档日期：2026-11-12；2026-11-13 更正指标口径并补充新版，见 §9

---

## 1. 总览

DSpark 是"并行 backbone + 轻量顺序头"的投机解码方案。本仓库实现包含三部分：

| 组件 | 作用 | 文件 |
| --- | --- | --- |
| **Target** | 被加速的 0.8B 主模型，全程冻结、只做 teacher | `Qwen3.5-0.8B-Base` |
| **Drafter** | 一次前向产出 7 个 draft token，供 target 批量验证 | `dflash_big.py::DFlashDrafterBig` |
| **Engine** | 拒绝采样验证、缓存快照回滚、置信度调度 | `dspark.py::DSparkEngine` |

核心数字：

| | 参数量 | 占 target |
| --- | --- | --- |
| Target | 873.4M（推理加载 853M） | 100% |
| **Drafter** | **130.70M** | **15.0%** |

---

## 2. Target 模型（Qwen3.5-0.8B-Base）

| 项 | 值 |
| --- | --- |
| 层数 | 24（18 层 Gated DeltaNet + 6 层 Gated Attention） |
| 全注意力层索引 | **3 / 7 / 11 / 15 / 19 / 23** |
| hidden | 1024 |
| FFN 中间维 | 3584 |
| 词表 | 248,320（embedding 与 lm_head 共享） |
| 注意力 | 8 Q 头 / 2 KV 头，head_dim 256 |
| 线性注意力 | 16 QK 头 + 16 V 头，head_dim 128，卷积核 4 |
| 原生上下文 | 262,144 |

**关键约束**：18 层线性注意力持有循环状态（`conv_states` + `recurrent_states`），`is_croppable == False`，**无法裁剪**。这决定了投机解码的回滚必须靠"深拷贝快照 + 重放"，而不能靠 crop KV。

---

## 3. Drafter 架构（DFlash 式）

### 3.1 数据流

```
target 隐状态（5 层 × 4 位置窗口）
        │                     ┌─────────────┐
        ├─ ctx_norm → ctx_fuse (5120→1024) ─┤ 融合后的 context (B,W,1024)
        │                                     │
bonus token 的 embedding ── in_norm/in_proj ─┤
        │                                     │
        └── anchor 向量 + 7 个 mask token → [X (B,8,1024)]
                                              │
                              因果/双向块注意力，每层注入 context 作为 K/V
                                              │
                        4 × DFBlock ──────────┤
                                              │
                        out_norm + out_proj → (B,7,1024)
                                              │
                     lm_head（共享 target 的）│
                                              │
                     + Markov 双词偏置 ───────┤
                                              │
                                        7 个位置的 logits
```

### 3.2 模块与参数量

| 模块 | 参数量 | 说明 |
| --- | --- | --- |
| `markov` | **63,569,920** | rank-128 双词表，`w1`/`w2` 各 [248320, 128] |
| `blocks`（4 层） | **59,777,024** | 每层 14,944,256 |
| `ctx_fuse` | 5,242,880 | 5120 → 1024 低秩压缩 |
| `in_proj` | 1,048,576 | 1024 → 1024 |
| `out_proj` | 1,048,576 | 1024 → 1024 |
| `mask_emb` | 7,168 | 7 个可学习槽位向量 |
| `ctx_norm` | 5,120 | RMSNorm(5120) |
| `anchor_emb` / `in_norm` / `out_norm` | 1,024 ×3 | |
| **合计** | **130,702,336** | **15.0% of target** |

**单层 DFBlock 细分（14.94M）**：

| 子模块 | 参数 | 作用 |
| --- | --- | --- |
| `q` / `k` / `v` / `o` | 1,048,576 ×4 | draft token 自身的注意力 |
| `k_ctx` / `v_ctx` | 1,048,576 ×2 | **每层独立的 context K/V 注入** |
| `gate` / `up` / `down` | 2,883,584 ×3 | SwiGLU，1024 → 2816 → 1024 |
| `norm1` / `norm2` | 1,024 ×2 | RMSNorm |

### 3.3 关键设计点

**① Target 特征取 5 层**

```python
CTX_LAYERS = (1, 7, 13, 19, 23)          # 层索引（0-based）
CTX_HS     = (2, 8, 14, 20, -1)          # 对应的 hidden_states 下标
```

`output_hidden_states` 会在最前面插入 embedding 输出，所以 `hidden_states[i]` 是**第 i−1 层**的输出，`hidden_states[-1]`（即下标 24）是过最终 RMSNorm 的流。取 `(1,7,13,19,23)` 而非直接用 `(3,15,23)` 的原因：后者会错位取到线性注意力层。

**② 每层独立 KV 注入**

每个 `DFBlock` 有自己的 `k_ctx` / `v_ctx`，把压缩后的 context 投影成 K/V，与 draft token 的 K/V 拼接：

```python
k = torch.cat([sp(self.k_ctx(ctx)), sp(self.k(h))], dim=2)
v = torch.cat([sp(self.v_ctx(ctx)), sp(self.v(h))], dim=2)
```

这样 context 对**每一层**都直接可见，而不是只在输入处拼一个前缀 token。

**③ 块内双向注意力**

```
anchor 行：可见 context
draft 行：可见 context + anchor + 全部 draft 槽位（双向）
```

设计意图是让 7 个 draft 位置一次并行产出。块内依赖由 Markov 头补回。

**④ Markov 顺序头（替代 GRU）**

```python
# DSpark 参考实现的双词偏置
bias = w1[prev_token] @ w2.T          # (V,)
logits = lm_head(z) + bias
```

| 项 | 值 | 说明 |
| --- | --- | --- |
| rank | **128** | 官方 256；因词表 248,320 是官方 151,936 的 1.63 倍，用 256 会使该头达 127M（14.5% of target），故降为 128 |
| 初始化 | w1 ~ N(0, 0.02)，w2 = 0 | 训练初期等价于无偏置 |
| 条件 | 位置 j 用**第 j−1 个 token** | slot 0 用 bonus token；slot j 用 x_{t+j} |
| 成本 | 一次查表 + 一次矩阵乘 | 无序列展开，比 GRU 快 |

**⑤ 共享词表**

Drafter 通过 `object.__setattr__` 持有 target 的 `embed_tokens` 与 `lm_head`，不注册为子模块：

- 词表零成本（否则 254M 参数）；
- 输出空间天然与 target 对齐；
- **不会**被写入 checkpoint，也不进优化器。

---

## 4. 训练配置

| 项 | 值 |
| --- | --- |
| 损失 | `KL(p_target ‖ q_draft)`，按位置加权 |
| 位置权重 | `[1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]` 归一化（越靠后权重越低） |
| 训练数据 | **target 自己生成的续写**（on-policy），600 条 × 512 token ≈ 307k token |
| 优化器 | AdamW，lr 1e-3，OneCycle（pct_start=0.03），weight_decay 0.01 |
| batch | B=64，K=8（1 bonus + 7 draft） |
| 步数 | 8000 |
| 精度 | drafter fp32 训练，teacher 用 bf16 + `no_grad` |
| 步时 | 75 ms（L20） |

**Target 冻结的强制措施**（代码级，非约定）：

```python
for p in model.parameters():
    p.requires_grad_(False)
model.eval()
# train() 开头还有一道断言：
teacher = {id(p) for p in model.parameters()}
leaked = [n for n, p in drafter.named_parameters() if id(p) in teacher]
if leaked:
    raise RuntimeError(f"target leaked into drafter: {leaked[:5]}")
```

已验证：目标权重文件 sha256 未变（`c2b1e5a1…adfbab6c`），mtime 停留在下载时刻；构造 drafter 前后逐位比对一致；target 与 drafter 参数交集为 0。

---

## 5. 训练结果

### 5.1 收敛曲线（8000 步）

| step | KL | top-1 | TV |
| --- | --- | --- | --- |
| 100 | 5.715 | 9.6% | 0.938 |
| 2000 | ~3.0 | ~30% | ~0.76 |
| 4500 | ~2.4 | ~44% | ~0.60 |
| 6000 | 1.141 | 61.5% | 0.482 |
| 8000 | **1.079** | **63.0%** | **0.469** |

### 5.2 逐位置准确率与 E[n]

Teacher-forced **argmax 一致率**（相当于 T=0；D=32，W=4 窗口）。E[n] 只数被接受的草稿 token，**不含 bonus token**，与论文的 τ 不可直接比，τ = E[n] + 1，见 §9.1：

| 数据集 | slot1 | slot2 | slot3 | slot4 | slot5 | slot6 | slot7 | **E[n]（不含 bonus）** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **on-policy（训练分布）** | 74.0% | 71.2% | 68.8% | 66.9% | 66.7% | 58.3% | 49.0% | **2.17** |
| 人写 Python | 45.0% | 40.8% | 32.5% | 35.8% | 42.5% | 23.3% | 16.7% | 0.73 |
| 散文 / 中文 | 11.1% | 9.4% | 9.4% | 8.3% | 6.7% | 7.8% | 7.2% | 0.12 |

**位置衰减已大幅缓解**：74% → 49% 跨 7 个位置（此前版本 61% → 0%）。

### 5.3 与官方对比

| | 官方（Qwen3-4B） | 本实现（Qwen3.5-0.8B） |
| --- | --- | --- |
| draft 参数占比 | 15.4%（615M） | **15.0%（131M）** |
| target 特征层数 | 5 | **5** |
| Markov rank | 256 | 128 |
| draft block | 7 | 7 |
| **τ（含 bonus）** | 未核实，见 §9.1 | **3.17**（= 2.17 + 1；T=0 的 argmax 估计，训练分布内） |
| 训练数据 | 数十亿 token | 307k token |

---

## 6. 已实现的其他组件

| 文件 | 内容 |
| --- | --- |
| `dspark_paper.py` | **现行 drafter**：整条前缀上下文、共享 K/V、置信度头、RoPE（对照论文重写） |
| `dspark_fast.py` | **现行推理引擎**：批量验证 + GDN 数学回滚 + 逆 CDF 采样（见下） |
| `dflash_big.py` | 早期 drafter（4 层、4 位置窗口；对照 `dspark_paper.py`） |
| `make_target_data.py` 等历史脚本 | 已随 `src4dspark/legacy/` 清理移除（2026-11-20） |

**采样层（2026-11-20 起）：全链路逆 CDF + 残差传播**

| 采样点 | 原（Gumbel-max，已删除） | 现（逆 CDF） |
| --- | --- | --- |
| anchor token（`_sample`） | `argmax(row/T + G)`，每处新抽噪声 | `u₀ ~ U(0,1)`，`searchsorted(p.cumsum(-1), u₀)` |
| draft slots（`_propose_multi/single`） | 每槽每 token 独立 Gumbel 噪声 | 首槽 u 来自 u₀（单链）或分层 `u_b=(π(b)+v_b)/B`（多分支；π 为每轮随机置换，缺了它分支 b 被钉死在第 b 层、提案分布有偏）；每消费一个 token 后 `u ← (u−lo)/q(x)` 残差传播到下一槽 |
| 对照基线 | — | `mode='iid'`：每条分支独立抽 `u_b ~ U(0,1)`，走同一条逆 CDF 链（与 `mode='stratified'` 仅随机数抽取方式不同，**不含 Gumbel**） |

**Gumbel-max 已从引擎中彻底删除**（2026-11-21）：`dspark_fast.py` 内不再有任何 Gumbel 噪声或 argmax 采样代码，全部采样点均为逆 CDF。

要点：**第一个 u 是随机的**（anchor 轮新抽 / 分支层分层均匀），**后续的 u 都是残差**——走进所选 token 的子区间后归一化回 [0,1)，仍精确均匀（残差引理）。条件分布 `q = softmax((U + B)/T)` 与接受判定完全不变，仅采样算子替换，无损性保持。这样做的动机：QMC 跨槽/跨分支耦合需要可传播的残差结构，Gumbel 噪声不具备（详见 `DSpark-多分支引擎.md` §3.7 与周报第 13 页）。

**Engine 的回滚策略**（因 GDN 状态不可裁剪；`dspark_fast.py`）：

- 全接受：直接提交接受前缀（1 次 target 前向/轮）；
- 部分接受：GDN 数学回滚（快照 + delta-rule 前缀结合律重建，~1.1ms），无第二次前向。

---

## 7. 已知短板

1. **训练语料单一**：prompt 全部采自 40MB Python 源码，导致散文/中文下 E[n] 仅 0.12。
2. **训练量偏小**：307k token 对 131M 参数偏少，存在过拟合风险。
3. **Markov rank 与官方不一致**（128 vs 256），原因是词表更大。
4. **推理 kernel 未优化**：单次前向 20.5ms 中 GPU 仅 4ms，瓶颈在 CPU 调度与 GDN 的 Python 逐头循环。当前即使 E[n]=2.17，端到端仍不划算；需先把单次前向压到 5ms 以内。

---

## 8. 后续计划（未执行）

1. 混合语料（散文 / 中文 / 对话 / 代码 / 数学）重新生成 on-policy 数据；
2. 数据量提升到 1.5M+ token（3000 条 × 512）；
3. 同超参重训，目标 E[n] ≥ 2.5；
4. 推理 kernel 优化（CUDA graph / 融合 kernel）；
5. 用 iid 采样而非分层采样重新评测接受率。

---

## 9. 更正与新版（2026-11-13）

对照论文原文（arXiv 2607.05147）重读后，本文需要更正或补充如下。

### 9.1 指标口径

- **τ 含 bonus token**（论文脚注 4）。§5 的 E[n] 只数草稿 token，可比的 τ = E[n] + 1：训练分布内 3.17，人写 Python 1.73，散文 / 中文 1.12。
- §5 的数字是 teacher-forced 的 **argmax 一致率**，相当于 T=0。论文用 T=1.0 的随机接受评测，接受概率是 Σmin(p, q)，两者不是同一个量。
- 所有数字都在 Markov 头以 Target 真实 token 为条件的前提下得到，**没有端到端验证**草稿用自己采样的 token 做条件时的表现。
- §5.3 里官方一栏的 τ≈4.96、615M / 15.4%、5 层特征，**在论文我已读的章节里找不到出处**，来自此前一轮的查询，未核实，不要当作论文结论引用。论文里确认过的只有：block 长度 7，Markov 头默认 rank 256，DSpark 和 DFlash 的 drafter 都是 5 层，训练用 Open-PerfectBlend 的 prompt、由 Target 重新生成 response、训练 10 个 epoch，评测温度 1.0。

### 9.2 旧版、论文、新版对照

| 项 | 论文 | 旧版 dflash_big.py | 新版 dspark_paper.py |
| --- | --- | --- | --- |
| 上下文 | anchor 前的整条序列（Eq. 2–3） | anchor 前 4 个位置 | 整条前缀，多个 anchor 用 mask 打包 |
| 层数 | 5 | 4 | 5 |
| 上下文 K/V 投影 | 与草稿行共享 W_K、W_V（Eq. 3） | 每层独立 k_ctx / v_ctx | 共享 |
| 融合顺序 | W_c 投影后 RMSNorm（Eq. 2） | RMSNorm 后投影 | 同论文 |
| 块内可见性 | 全部位置双向可见 | anchor 行只看自己和上下文 | 全部 8 行双向可见 |
| Markov rank | 256 | 128 | 128（词表更大，有意降低） |
| 损失 | 0.1·CE + 0.9·TV + 1.0·BCE，w_k = exp(−(k−1)/γ)，不归一化 | KL，手工权重，归一化 | 同论文 |
| 置信度头 | 有，共享 Markov embedding | 无 | 有 |
| 位置编码 | 未写 | 无 | RoPE（我的补充） |
| anchor 行输出 | anchor 本身出第 1 个草稿 logits（γ 个输入出 γ 个 logits） | 丢弃 | **未改**：仍是 anchor + 7 个 mask、丢弃 anchor 输出；论文称质量相近、计算更少 |
| 训练量 | Open-PerfectBlend 的 prompt（未说明是否抽子集），10 个 epoch | 约 307k token | 5 万条 prompt × 10 epoch |
| STS 校准、前缀调度器 | 有 | 无 | 无 |

### 9.3 新版状态

- 参数量 135,157,889（15.47% of target）。
- 只做过随机 token 的冒烟测试：前向、反传正常；Target 参数无梯度；mask 隔离已验证（改动 anchor 之后的上下文和另一个 block 的 bonus，anchor 0 的输出变化为 0.0）。
- 正式训练已完成：10 个 epoch，30,910 步（5 个 epoch 后续训到 10 个），Target 权重文件 sha256 前后一致。验证集 tau 从随机初始化的 1.005 升到 1.806（逐位置 alpha 0.50 / 0.41 / 0.36 / 0.32 / 0.30 / 0.30 / 0.29），CE 12.48 → 4.47，没有过拟合。5 个 epoch 时为 1.738，续训到 10 个 epoch 又提升 0.068。权重在 `dflash_pb.pt` / `dflash_pb_best.pt`，曲线图 `doc/dspark_training_curves.png`。完整结果见 `DSpark-训练文档.md` §8。旧版 `dflash_big.pt` 在同一批 block 上 tau 为 1.152，但两者训练数据、上下文、温度都不同，不是严格消融。
- `dspark_pb_train.ipynb` 未运行，实际训练走的是 `pb_train.py` + `pb_after_gen.sh`。
- 温度：生成、损失里的 p_t 和 q、验证共用同一个 TEMP。这是我的设计选择，论文没有规定损失里的温度，也没有规定训练 response 的采样参数。
- 学习率、优化器、batch 论文没写，沿用此前的 AdamW + OneCycle，lr 1e-3。

### 9.4 现行文件

| 文件 | 作用 |
| --- | --- |
| `dspark_paper.py` | 新版 drafter：整条前缀上下文、共享 K/V、置信度头、RoPE |
| `dspark_fast.py` | 推理引擎：批量验证、GDN 数学回滚、**全链路逆 CDF + 残差采样**（2026-11-20 anchor 采样从 Gumbel-max 改为逆 CDF） |
| `dflash_big.py` | 早期 drafter（4 层、4 位置窗口）；`_sample` 对照用 |
| `dspark_loss.py` | 取块、损失、评测 |
| `pb_gen_worker.py` | 每卡一个进程，Target 纯温度采样生成 response，可断点续跑 |
| `pb_train.py` | 后台训练脚本，指标写入 `train_log.jsonl` |
| `pb_check_gen.py`、`pb_after_gen.sh` | 校验生成数据是否齐全；生成结束后自动启动训练 |
| `dspark_plot.py`、`dspark_pb_live.ipynb` | 实时画训练曲线的模块和 notebook |

> `src4dspark/legacy/` 和 `dspark_e2e.py` 已于 2026-11-20 清理移除。
