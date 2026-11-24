# DSpark 训练文档（Qwen3.5-0.8B-Base）

> 状态：10 个 epoch 已训完（2026-11-13），验证集 tau 从 1.005 升到 1.806，Target 权重未变，结果见 §8。
> 日期：2026-11-13
> 论文：DSpark（arXiv 2607.05147）。模型和损失与论文的逐项对照见 `DSpark-架构与参数.md` §9

### 文件布局（2026-11-13 整理）

| 目录 | 内容 |
| --- | --- |
| `/home/dev/KAP/src4dspark/` | DSpark 模型与推理源码：`dspark_paper.py`（现行 drafter）、`dflash_big.py`（基础模块）、`dspark_e2e.py`（端到端解码）；`legacy/` 为早期实验归档 |
| `/home/dev/KAP/train/` | 训练 drafter 的源码：生成、校验、训练、损失、STS、监控脚本和 notebook |
| `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original/` | 原始 target 的只读拷贝，sha256 校验一致；所有脚本默认从这里加载 target |
| `/home/dev/KAP/parameter/draft/dspark_pb/` | 现行 drafter 权重、续训状态、STS 温度 |
| `/home/dev/KAP/parameter/draft/legacy/` | 早期 drafter 权重 |
| `/home/dev/KAP/data/pb_dspark/` | prompt、生成的 response、训练日志 |
| `/home/dev/KAP/test/dspark_latency.ipynb` | 每 token 耗时对比 |

每个目录下有 README.md。

---

## 1. 流程总览

```
Open-PerfectBlend（1,420,909 条对话）
  └─ 抽 50,500 条 prompt（只取第一轮 human，数据集自带的 response 不读）
       └─ Target（Qwen3.5-0.8B-Base）以 T=1.0 纯温度采样续写 response
            └─ 训练 DSparkDrafter：Target 全程冻结，只更新 drafter 的参数
                 └─ 验证集评估 tau 和逐位置接受率 → train_log.jsonl → notebook 实时画图
```

| 阶段 | 脚本（`/home/dev/KAP/train/`） | 产物（`/home/dev/KAP/data/pb_dspark/`，另注明的除外） |
| --- | --- | --- |
| 抽 prompt | `dspark_pb_train.ipynb` §1 | `prompts.pt`、`prompts_raw.jsonl` |
| 生成 response | `pb_gen_worker.py`，每张 GPU 一个进程 | `gen/shard_r*_*.pt` |
| 校验数据是否齐全 | `pb_check_gen.py` | `pipeline.log` |
| 训练 | `pb_train.py` | `train_log.jsonl`、`train_stdout.txt`；`train_state.pt` 在 `parameter/draft/dspark_pb/` |
| 自动衔接 | `pb_after_gen.sh`：等生成进程退出 → 校验 → 启动训练 | `pipeline.log` |
| 实时监控 | `dspark_pb_live.ipynb` + `dspark_plot.py` | 只读 `train_log.jsonl` |

权重存到 `/home/dev/KAP/parameter/draft/dspark_pb/dflash_pb.pt`（最近一次验证点）和 `dflash_pb_best.pt`（验证 tau 最高的一次），不覆盖旧的 `dflash_big.pt`（现在在 `parameter/draft/legacy/`）。

---

## 2. 数据

### 2.1 prompt

- 来源：Open-PerfectBlend，6 个 parquet 分片，1,420,909 条，8 个来源。
- 随机种子 0。只取第一轮 human 的内容，要求非空、套完 chat template 后不超过 1024 token，并做精确去重。
- 共 50,500 条：训练 50,000 条，验证 500 条。
- prompt 长度（含 chat template）p50 / p90 / p99 / max = 78 / 275 / 775 / 1024 token。
- 来源构成：MetaMathQA 13,001；UltraInteract 9,090；ultrachat 7,755；orca-math 7,559；ultrafeedback 4,537；evol-codealpaca 4,173；AutoIF 2,261；lmsys-arena 2,124。
- 渲染方式：Qwen3.5 的 chat template，加生成提示（含空的 think 块），Base 模型直接续写。

### 2.2 response（由 Target 自己生成）

- 温度 `TEMP = 1.0`，并且是纯温度采样：`top_p=1.0, top_k=0`，样本严格服从 `softmax(logits / TEMP)`。模型目录里没有 `generation_config.json`，所以 worker 里显式传参。
- 最多 512 个新 token，遇到 `<|im_end|>` 或 `<|endoftext|>` 停止。
- 没有 vLLM，用 HF `generate`，每卡 batch 128，OOM 时自动降到 64、32。按 prompt 长度升序处理，所以最长的 prompt 最后才生成。
- 吞吐：每卡约 2,900–3,700 tok/s（生成日志里的累计值）。
- 冒烟样本（512 条）统计：平均 389.5 token；43.6% 自然结束，56.4% 顶到 512 上限被截断；distinct-4gram 均值 0.891，没有大面积重复循环。
- 开放域问答在 T=1.0 下确实比较发散，会出现编造的词。drafter 学的就是 Target 在这个温度下的分布，所以保留。
- 训练时只保留 response 不少于 16 token 的样本。

---

## 3. 模型

`dspark_paper.py::DSparkDrafter`，135,157,889 个参数，占 Target 的 15.47%。

- 上下文：anchor 之前整条序列的 5 层 Target 特征（层索引 1、7、13、19、23），`H_ctx = RMSNorm(W_c [H^(l1);…;H^(l5)])`，注入每个草稿层的 K/V。
- 5 层，hidden 1024，8 头，FFN 2816；上下文行和草稿行共享 W_K、W_V（论文式 3）。
- block 内 8 行（anchor + 7 个 mask）双向可见；同一条序列上的多个 anchor 打包进一次前向，用 mask 保证每个 anchor 只看自己位置之前的上下文和自己的 block。
- Markov 头 rank 128（论文 256，因词表更大而降低）；置信度头 `c_k = σ(wᵀ[h_k; W1[x_(k-1)]])`，共享 Markov embedding。
- RoPE 加在 q/k 上。论文没写位置编码，这是我的补充。
- embedding 和 lm_head 与 Target 共享且冻结，不进 checkpoint，也不进优化器。

---

## 4. 损失函数

对每个 block（1 个 bonus token + 7 个 Target 采样出的 token），k = 1..7：

```
L = Σ_k w_k · (0.1·CE_k + 0.9·TV_k + 1.0·BCE_k),   w_k = exp(−(k−1)/7)
```

- `CE_k = −log q_k(x*_k)`，x*_k 是 Target 实际采样出的 token。
- `TV_k = ‖q_k − p_k‖₁`。p = softmax(Target logits / TEMP)，q = softmax(draft logits / TEMP)，用同一个 TEMP。
- `BCE_k`：置信度头对 `c*_k = 1 − TV_k/2` 的二元交叉熵，c* 只当标签，不反传。
- 位置权重 w = 1.00 / 0.87 / 0.75 / 0.65 / 0.56 / 0.49 / 0.42，不做归一化，对 block 取平均。
- Markov 头训练时的条件 token 是 Target 采样出的真实 token（teacher forcing）。论文没写这一点，沿用此前做法。

监控指标：

- `alpha_k = Σ min(p, q) = 1 − TV_k/2`，是第 k 位在 T=TEMP 下的接受概率。
- `tau = Σ_k Π_(j≤k) alpha_j + 1`，加 1 是 bonus token（论文脚注 4）。这是 teacher-forced 估计，不是端到端实测。
- 随机初始化时 CE ≈ 12.5（接近 ln 248320 ≈ 12.4），TV ≈ 2，tau ≈ 1，这是曲线的起点。

## 5. 训练配置

| 项 | 值 |
| --- | --- |
| 优化器 | AdamW，weight_decay 0.01 |
| 学习率 | 峰值 1e-3，OneCycle（pct_start 0.03）；梯度裁剪 1.0 |
| 超参来源 | 学习率、优化器、batch 论文都没写，沿用此前的设置，没有调过 |
| epoch | 5（论文 10）。1 个 epoch = 训练序列数 ÷ 16 步，约 3,125 步，5 个 epoch 约 15,625 步 |
| 每步 | 16 条序列 × 每条 8 个 anchor = 128 个 block |
| anchor | 在每条序列的 response 区域内随机取，每步重新抽 |
| 精度 | drafter fp32（开 TF32），Target bf16 + no_grad |
| 验证 | 每 500 步一次，用 200 条验证序列，固定随机种子 |
| 存盘 | 每次验证后存 `dflash_pb.pt` 和 `train_state.pt`；验证 tau 创新高时另存 `dflash_pb_best.pt` |
| 断点续训 | 重新运行 `pb_train.py` 会读 `train_state.pt` 接着跑 |
| GPU | 训练用 GPU 0 |

手动启动（一般不需要，`pb_after_gen.sh` 会自动启动）：

```bash
cd /home/dev
CUDA_VISIBLE_DEVICES=0 /home/dev/venv/bin/python -I /home/dev/KAP/train/pb_train.py --epochs 5 --temp 1.0
```

---

## 6. Target 冻结

- Target 所有参数 `requires_grad=False`，并且 `model.eval()`。
- drafter 持有的 embedding 和 lm_head 是 Target 的同一份对象，用 `object.__setattr__` 挂上，不注册为子模块，所以不进 `state_dict`，也不进优化器。
- 启动时断言 Target 参数和 drafter 参数的交集为空。
- 优化器只拿 drafter 的参数。
- 训练前后各算一次 Target 权重文件的 sha256，写进 `train_log.jsonl` 的 `config` 和 `done` 记录。试跑的结果是前后一致：`c2b1e5a1…adfbab6c`。
- sha256 只能证明磁盘上的文件没变。内存里的参数没被改，靠的是上面的 `requires_grad` 和优化器范围。

---

## 7. 怎么看训练在变好还是变差

`dspark_pb_live.ipynb` 只读 `train_log.jsonl`，每 30 秒重画一次，训练结束后自动停；中断这个 cell 不影响训练。

| 图 | 变好的样子 | 变差的信号 |
| --- | --- | --- |
| 训练 loss | 持续下降 | 上升或剧烈震荡 |
| tau（train / val） | 两条都上升 | val 下降 |
| TV、CE（train / val） | 都下降 | train 降、val 升或持平，是过拟合 |
| 逐位置 alpha | 蓝线整体高于灰色虚线（起点），后面的位置衰减得更慢 | 蓝线贴着起点 |
| 置信度头 vs 实际 alpha | 两条线接近 | 置信度头系统性偏高，是过度自信 |

图顶部一行是进度和预计剩余时间，图上方还有一行文字判断，比较 val tau 的起点、当前值和上一次验证值。验证 tau 连续 3 次下降会给出警告。

---

## 8. 结果

### 8.1 运行信息

- 生成：2026-11-13 10:11 开始，11:21 结束，50,500 条 response 全部到齐。
- 训练分两段：第一段 5 个 epoch（11:22:16–12:44:59，15,455 步）；第二段从 15,455 步续训到 10 个 epoch（合计 30,910 步，约 82 分钟）。两段都以退出码 0 结束。
- 约 318 ms/步。
- 训练集 49,457 条序列，验证集 496 条（response 不足 16 token 的被丢弃）。
- 参数 135,157,889。
- Target 权重文件 sha256 训练前后一致：`c2b1e5a1…adfbab6c`。
- 权重：`dflash_pb.pt`（最后一个验证点）、`dflash_pb_best.pt`（验证 tau 最高，step 30910）。
- 曲线图：`dspark_training_curves.png`。
- 续训时把 `--epochs` 从 5 改成 10，学习率调度按 30,910 步重建并快进到 15,455 步，所以第二段开头学习率重新抬起来，验证 tau 从 1.710 短暂回落到 1.677，随后随学习率退火回升。

### 8.2 训练过程

| step | val tau | val CE | val TV |
| --- | --- | --- | --- |
| 0（随机初始化） | 1.005 | 12.482 | 1.990 |
| 500 | 1.271 | 6.283 | 1.693 |
| 1500 | 1.399 | 5.672 | 1.544 |
| 3000 | 1.486 | 5.468 | 1.477 |
| 5000 | 1.540 | 5.177 | 1.451 |
| 10000 | 1.646 | 4.839 | 1.384 |
| 15455（5 个 epoch） | 1.710 | 4.761 | 1.346 |
| 20000 | 1.707 | 4.763 | 1.346 |
| 25000 | 1.763 | 4.639 | 1.316 |
| 30910（10 个 epoch） | **1.777** | 4.608 | 1.308 |

- 上升最快的阶段是前 1,500 步，约占总提升的 51%。之后增速逐渐变慢，但到最后一个验证点仍在上升（30500 → 30910 是 +0.000，末尾已走平）。
- 15,455 → 20,000 步有一次回落（1.710 → 1.707），是续训时学习率被重新抬高造成的，随后随学习率退火恢复并超过原来的水平。
- 训练集 tau 1.820、验证 1.777，训练集 loss 10.00，验证集各项与训练集的差距在 0.04–0.15 之内。**没有看到过拟合**。
- 置信度头几乎与实际 alpha 重合（见曲线图右下），ECE 没有单独测，STS 校准没有做。

### 8.3 最终验证集结果（最佳权重，496 条序列）

| 指标 | 新 drafter（10 epoch） | 旧 `dflash_big.pt` |
| --- | --- | --- |
| tau | **1.806** | 1.152 |
| tau_argmax（T=0） | 1.819 | — |
| CE | 4.474 | — |
| TV | 1.296 | — |
| 逐位置 alpha | 0.50 / 0.41 / 0.36 / 0.32 / 0.30 / 0.30 / 0.29 | 0.13 / 0.12 / 0.12 / 0.11 / 0.11 / 0.11 / 0.11 |

注意：评估用的 block 是重新随机抽的（A=8，种子 123），和训练期验证的 1.777 不是同一批，所以 1.806 略高。

### 8.4 按来源拆分

| 来源 | n | tau 新 | tau 旧 |
| --- | --- | --- | --- |
| MetaMathQA | 152 | 2.14 | 1.17 |
| orca-math | 63 | 2.07 | 1.16 |
| UltraInteract | 76 | 2.00 | 1.20 |
| evol-codealpaca | 50 | 1.59 | 1.17 |
| ultrafeedback | 32 | 1.55 | 1.13 |
| AutoIF | 24 | 1.53 | 1.17 |
| lmsys-arena | 23 | 1.51 | 1.11 |
| ultrachat | 76 | 1.48 | 1.10 |

数学类最高，开放对话最低，和论文的规律一致（结构化文本接受率高，聊天低）。所有来源都比旧版好，但两者训练数据、上下文、温度都不同，这不是严格消融。

### 8.5 结论

- **训练是有效的**：tau 从随机初始化的 1.005 升到 1.806，CE 从 12.48 降到 4.47，两段训练里曲线都朝对的方向走，没有过拟合。10 个 epoch 比 5 个 epoch 多出 0.068（1.738 → 1.806）。
- **但绝对水平仍然低**：第 1 位 alpha 0.50，第 7 位 0.29。按 τ = 1.81 算，一次验证最多多换 0.81 个 token 的收益，而两次 Target 前向的成本（GDN 状态不可裁剪）在 batch=1 下仍然不划算。这和之前对推理 kernel 的结论一致。
- 10 个 epoch 末端提升已经很小（最后 6 个验证点只涨 0.002），再训收益有限。主要瓶颈已经不是训练轮数，而是训练数据量（论文用全量 prompt 训 10 个 epoch，这里 5 万条）和推理 kernel。

### 8.6 训练期间一处没有踩到的问题

`pb_after_gen.sh` 用 `pgrep -f` 按生成脚本的文件名判断"生成是否结束"。我在检查进度时运行的命令里带有同一个文件名，会被它当成还在跑的生成进程，导致训练启动被推迟了约半小时。数据没受影响，但以后要跑类似的等待脚本，最好改成等具体 PID，或者把要匹配的字符串拆开写。


---

## 9. 端到端每 token 耗时对比

已实测，见 `dspark_latency.ipynb`（已执行，输出和图表都在里面）。同一模型、同一批 prompt、温度 1.0、每条最多 96 token，只计 decode，prefill 排除，结果取 6 条 prompt 的中位数。

| | baseline（直接推理） | DSpark |
| --- | --- | --- |
| 每 token 耗时 | **17.46 ms** | **22.78 ms** |
| 比值（baseline / DSpark） | 0.766，即 DSpark 慢 31% | |
| 每轮接受草稿数 | — | 0.83 |
| 每轮 Target 前向次数 | — | 2.00 |

分 prompt 看：

| prompt | baseline | DSpark | 比值 | 接受/轮 |
| --- | --- | --- | --- | --- |
| 17 * 24 | 16.74 | 21.66 | 0.77 | 1.00 |
| 天空为什么蓝 | 17.64 | 27.91 | 0.63 | 0.29 |
| Python 反转字符串 | 17.60 | 18.78 | 0.94 | 1.43 |
| 三种水果颜色 | 17.33 | 23.90 | 0.73 | 0.67 |
| 火车速度题 | 17.58 | 18.33 | 0.96 | 1.48 |
| 罗密欧与朱丽叶 | 17.30 | 28.33 | 0.61 | 0.26 |

**为什么还是慢**：
- 6 条 prompt、221 轮里**没有一轮 7 个草稿全被接受**，所以每轮固定 2 次 Target 前向（1 次验证 + 1 次回滚重放）。
- 平均每轮接受 0.83 个草稿，一轮产出 1.83 个 token，用掉 2 次前向。baseline 是 1 次前向出 1 个 token，所以 DSpark 每 token 的前向次数是 `2 / 1.83 = 1.09`，比 baseline 多 9%。
- 时间分配：drafter 占 8%，验证 46%，回滚重放 46%。drafter 本身很便宜，成本几乎都在 Target 的两次前向上。
- 平价条件：`接受/轮 ≥ 前向次数/轮 − 1 = 1.00`。目前 0.83，差一点。
- 单次 Target 前向约 8–9 ms，其中 GPU 只占约 4 ms，其余是 CPU 调度开销。baseline 的 17.46 ms/token 里也有同样的开销。

**和离线评测的差距**：离线 teacher-forced 的 tau 是 1.806（E[n]≈0.81），端到端实测接受/轮 0.83，两者接近。teacher-forced 估计在这批 prompt 上是准的。

**结论**：在当前实现下，DSpark 比直接推理慢 31%。要让它真正获益，至少需要做到下列一条：
1. 单次 Target 前向压到 5 ms 以内（CUDA graph 或融合 kernel）；
2. 提高接受率到 1.5 以上（更多训练数据、更长的 block），摊薄每轮的前向次数；
3. 解决 GDN 状态不可裁剪的问题，省掉回滚重放的那一次前向。

---

## 10. 时间估计

- 生成：每卡累计吞吐约 2,900–3,700 tok/s。prompt 按长度升序处理，后面的 batch 会更慢，所以总耗时没法精确估计，进度看 `gen/log_r*.txt`。
- 训练：全程约 15,625 步。随机 token 的探测是 0.29 s/步（典型长度），试跑是 0.45–0.58 s/步（那时 GPU 0 还被生成进程占着），所以约 1.3–2.5 小时，不含验证和存盘。每次存盘约 2.2 GB（权重 540 MB + 优化器状态 1.6 GB）。

---

## 11. 已知局限

- tau 是 teacher-forced 估计。端到端实测（`dspark_latency.ipynb`）的接受/轮 0.83 与它接近，但那是 6 条 prompt 的小样本。
- 验证集和训练集来自同一个 prompt 来源，也是同一个生成方式，没有域外评测。
- T=1.0 下开放域问答的文本比较发散，这类内容上的 alpha 天然偏低。
- 冒烟样本里约 56% 的 response 顶到 512 上限被截断，drafter 很少见到长回答的结尾。
- 只训 5 个 epoch，论文是 10 个。
- RoPE 是我加的，论文没写位置编码。
- STS 校准和前缀调度器没有实现。


---

## 12. 与论文/官方实现的差距分析（2026-11-13）

问题：论文在 Qwen3-4B 上 τ 达到 3.5–6.1（GSM8K 6.11 / MATH 5.70 / MBPP 5.13 / Alpaca 3.54），位置 1 接受率约 0.72（chat）/0.88（math）；我的 10-epoch 训练 τ 只有 1.806、α₁=0.50。逐项核对论文原文（Dspark.txt）和官方源码（deepseek-ai/DeepSpec，`config/dspark/dspark_qwen3_4b.py`、`deepspec/modeling/dspark/loss.py` 等）后，差距按证据强度列出如下。

### 12.1 有定量实验证据的因素

**(a) 训练文本的分布太"平"（最大因素之一）** — `test/gap_entropy.py`，结果 `data/pb_dspark/gap_entropy.json`。

| 条件 | target 熵 H | pmax | 可达 α₁（重加权估计） |
| --- | --- | --- | --- |
| A base，T=1.0 纯采样（我原来的数据） | 1.393 | 0.681 | 0.483 |
| B base，T=0.7/p0.8/k20（官方参数） | 0.846 | — | 0.539 |
| C instruct，T=1.0 | 1.048 | — | 0.516 |
| D instruct，官方参数 | 0.790 | — | 0.549 |

论文原文："responses are regenerated by each target model with **recommended sampling parameters**"，官方 `prepare_data.sh`：`temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096`。我原来用 T=1.0 纯采样，是熵最大的生成方式。损失和评测仍是 T=1.0（官方对 cached logits 直接 softmax，评测温度 1.0）——**变的只是训练文本本身的熵**。同一模型仅换采样参数，可达 α₁ 估计 +0.056；换 instruct 目标再 +0.033。

**(b) 每个序列的 anchor 数（监督密度）** — `test/gap_anchors.py`，从零训练 2000 步、其余全部相同（16 seqs/步、同 lr/seed/数据顺序）：

| anchors/序列 | val τ | α₁ |
| --- | --- | --- |
| 8（我原来的设置） | 1.408 | 0.31 |
| 32 | **1.591** | 0.41 |

官方 `num_anchors=512`（≈每个 response 位置都监督），我原来 8（约 3% 位置）。仅 8→32 就 +0.18 τ；512 的收益会更大。

**(c) 优化器配方**（官方 `dspark_qwen3_4b.py` + `utils/optim.py`，已核对源码）：

| 项 | 官方 | 我原来 |
| --- | --- | --- |
| loss 衰减 γ | **4.0**（`exp(-(k-1)/4)`，代码里 `exp(-positions/γ)`） | 7 |
| loss 归一化 | **每项按权重和归一化**（加权平均 `num/den`） | 位置求和后对 block 取平均（尺度大 ~Σw≈6 倍，与 grad clip 1.0 相互作用） |
| lr / 调度 | 6e-4，cosine + 4% 线性 warmup，wd=0 | 1e-3，OneCycle，wd=0.01 |
| 精度 | bf16 模型 + fp32 master 权重 | drafter 纯 fp32 |

loss 的其余部分核对结果：L1 项官方就是 `Σ|p−q|`（=2·TV，不除 2，我的实现一致）；CE 目标是序列真实 token（一致）；BCE 目标是 `Σmin(p,q)` detach（一致）；权重 0.1/0.9/1.0（一致）。

### 12.2 没有单独定量、但量级上明确的因素

- **数据规模**：官方 1.24M prompts（Open-PerfectBlend 95% 训练集，全部 user turn 重新生成，max_length 4096）× 10 epochs，global batch 512 序列/步 → 每 epoch 约 6 亿 block 访问；我 5 万 prompts（第一轮 user turn）× 512 max_new × 10 epochs，16 序列/步 → 每 epoch 约 40 万。**约 3 个数量级**。
- **target 本身**：论文 Qwen3-4B（instruct），我 Qwen3.5-0.8B（base）。模型越小 T=1.0 分布越平（上面 C vs D 可见）；base 模型的开放域文本天然更发散。
- **Markov rank**：官方 256，我 128（有意降，词表 248k 是官方 152k 的 1.63 倍，rank 256 会到 ~199M/22.7%）。
- **anchor 行约定**：论文把 anchor 本身作为第一个预测位置（γ 输入出 γ logits），我是 anchor + 7 个 mask、丢弃 anchor 输出。论文自己说两种"质量相近"，是算力差异不是质量差异，暂不改。
- **多轮对话重新生成**：官方把每个 conversation 的所有 user turn 都重新生成；我只用第一轮。

### 12.3 结论

论文的高接受率 = 尖锐的训练分布（官方采样参数 + instruct 4B）× 高监督密度（512 anchors/序列）× 大数据量（3 个数量级）× 官方优化配方，四类因素叠加。我的 1.806 主要是"低监督密度 + 高熵训练分布 + 小优化配方偏差"的结果，不是架构问题（架构与官方逐模块一致）。

### 12.4 v2 重训（按官方配方，2026-11-13 启动）

在可承受的算力内对齐以上各项，脚本 `train/pb_train_v2.py`，流水线 `train/pb_pipeline_v2.sh`：

1. 重新生成 50,500 条 prompt 的 response：T=0.7/top_p0.8/top_k20，max_new 1024（官方 4096 的折中），目录 `data/pb_dspark/gen07/`；旧的 T=1.0 生成保留用于论文口径评测（τ_paper）。
2. loss：γ=4、每项加权平均；AdamW 6e-4、cosine + 4% warmup、wd 0、clip 1.0；bf16 autocast（fp32 master）。
3. anchors 32/序列（显存实测 28.7 GB @ 512 blocks/步）。两个并行运行：**A** = 32×16（512 blocks/步，10 epochs = 31,250 步），**B** = 64×8（62,500 步，测试更高覆盖）。
4. 评测口径：τ_paper（T=1.0 生成 response、T=1.0，与之前 1.806 直接可比）+ τ_indist（0.7 生成 response）。
5. 权重写 `parameter/draft/dspark_pb_v2_{a,b}/`，不动旧目录。

仍然对不上的项（算力限制）：数据量（5万 vs 124万 prompts）、max_new（1024 vs 4096）、单轮 vs 多轮、512 vs 32/64 anchors。若 v2 τ 明显上升，下一步按性价比补数据量。

### 12.5 v2 结果（2026-11-14）

| 运行 | 配置 | 步数 | τ_paper | τ_indist | α₁ / α₇ |
| --- | --- | --- | --- | --- | --- |
| 第一版（对照） | 8 anchors×16 序列，T=1.0 数据，γ=7 求和 loss，OneCycle 1e-3 | 30,910 | 1.806 | — | 0.50 / 0.29 |
| **v2-A** | 32×16，0.7 数据，γ=4 加权平均，6e-4 cosine，bf16 | 30,860（完成） | **2.133** | 2.528 | 0.61 / 0.32 |
| **v2-B** | 64×8，同上 | 61,710（进行中，@48k） | 2.134–2.138 | 2.555–2.561 | 0.61 / 0.32 |

- 官方配方三项改动（采样参数 0.7/0.8/20、anchor 密度 8→32、loss/优化器对齐）合计把 τ_paper 从 1.806 提到 **2.133**（+0.33），α₁ 0.50→0.61；训练同分布 τ 达 2.53。
- A/B 两个 anchor 配置几乎收敛到同一水平（2.133 vs ~2.14）：在 512 blocks/步 的总量下，每序列 32 或 64 个 anchor 的差别已经不大，瓶颈转移到数据量（5 万 vs 124 万 prompts）与 target 本身（0.8B base）。
- 运行/监控/评测 notebook：`train/dspark_v2_run.ipynb`（含启动命令、曲线图、checkpoint 复测 cell，评测 cell 已验证与训练日志一致）。
### 9.1 v2-A 的端到端加速实测（2026-11-14）

`test/dspark_v2_speedup.ipynb`（已执行）：12 条 val prompt × 256 token，GPU 0，prefill 不计时，v2-A best（teacher-forced τ_paper 2.133）。

| 条件 | baseline tok/s | DSpark tok/s | 加速比 | 端到端 τ（token/轮） | 草稿接受/轮 | 全接受轮占比 |
| --- | --- | --- | --- | --- | --- | --- |
| T=1.0（论文口径） | 57.06 | 52.98 | **1.08×** | 3.63 | 1.65 | 0.02 |
| T=0.7/0.8/20 | 56.41 | 56.16 | **1.00×** | 3.78 | 1.78 | — |

每轮 68.4 ms 分解（T=1.0）：drafter **5.3 ms** + 验证 **29.3 ms** + 回滚重放 **28.6 ms** + 采样/deepcopy ~5 ms。

对比第一版（§9）：接受/轮 0.83 → 1.65，每 token 22.78 ms → 18.8 ms，从"慢 31%"变成"**基本打平（+8%）**"。drafter 变快又变强之后，瓶颈完全暴露在引擎侧：98% 的轮次有拒绝 → 每轮固定 2 次 Target 前向（验证 + 重放）共 ~58 ms，占每轮 85%。若省掉重放（GDN 状态可裁剪或全接受），理论加速 ~1.7×；再把验证前向压到普通 decode 步的耗时，理论 ~2.8×。端到端 τ 3.63 高于 teacher-forced 2.133：前者把拒绝轮的 correction token 也计入了产出。

### 9.2 怎么省掉回滚重放（2026-11-14 审计结论，全部实测）

两个并行审计（Qwen3.5 GDN 实现 + 引擎逐行开销）的关键事实：

**缓存的真实构成**（ctx~500 实测，共 25.9 MB）：18 层 GDN 循环状态 fp32 共 18.9 MB + conv 状态 0.9 MB（与上下文长度无关）；6 层全注意力 KV 仅 ~12 kB/token，**逐 token 追加、可裁剪**。deepcopy 整个缓存 1.63 ms/轮且无条件执行（全接受轮纯浪费）；只快照 GDN 的替代方案实测 0.21+0.19 ms，logits 逐位一致。

**两个 target 前向都是 CPU 发射瓶颈**：8-token 前向 1798 次 kernel 发射，wall 29–36 ms 中 GPU 只忙 **5.8 ms**；单 token 前向 16.7–19.8 ms。torch.compile/CUDA-graph 是最大的单杠杆（同时也会加快 baseline，需按比值评估）。

**GDN 状态可以数学重建（已验证）**：delta-rule 更新对前缀满足结合律；从 verify 前向已收集的 hidden_states 重算每 token 的 q/k/v/β/g 因子（5 个 GEMM/层），用 fla `fused_recurrent_gated_delta_rule` 以 prefill 状态为初始值、**18 层一次 batched 调用**（布局 [18,16,128,128] 已实测）即可得到已接受前缀任意位置的循环状态——与真实前向最大误差 2.6e-3（bf16 conv 噪声级别）。陷阱：conv 窗口有 off-by-one（新位置 t 的窗口是 full_conv[t+1:t+5]）。HF 自带的 crop 只回退 conv、明确不回退循环状态。

**结论方案（按性价比排序）**：
1. Tier 1（半天）：GDN-only 快照替换 deepcopy、重放 logits_to_keep=1、向量化 _accept（每轮 12 次 CPU-GPU sync → 2 次）、drafter rope/mask 缓存 → ~60 ms/轮，~1.2×
2. Tier 2（1–2 天，核心）：数学回滚 + 把 correction token 直接当作下一轮的 anchor（它本来就是 target 分布的合法采样），verify chunk = [corr, d1..d7]——**每轮只剩 1 次 target 前向，重放前向彻底消失** → ~41 ms/轮，~1.5×
3. Tier 3（1 天）：torch.compile/CUDA-graph target 前向 → 每轮 ~13 ms → **~1.8–2.2×**（baseline 同步变快，按比值算）
4. 正交杠杆：τ_eff（数据扩量重训）——每轮产出的 token 数直接乘进加速比

审计遗留：基准里 draft_ms 只计了发射（drafter 的 GPU 时间被 verify 计时器吸收），真实的分段归因以本节为准。审计脚本 /tmp/dspark_audit_bench*.py。

### 9.3 快速引擎：数学回滚 + 每轮 1 次 target 前向（2026-11-14 实现）

`src4dspark/dspark_fast.py`（对照实现 `dspark_e2e.py` 保留）。三个改动：**① 拒绝时不再"快照+重放前向"，用 gated delta rule 的前缀结合律数学重建 18 层 GDN 状态**（一次 batched fla `fused_recurrent` 调用，实测 1.1 ms；q/k/v/β/g 因子从 verify 前向已收集的 hidden_states 在层外重算，conv 重建精确到 0.0）；**② correction token 作为下一轮的 anchor**（ lawful target sample），verify chunk 恒为 [anchor, d1..d7]；**③ 只快照 GDN 状态（0.2 ms），注意力 KV 拒绝时直接 crop，接受判定向量化**。

**无损性验证**（`test/test_dspark_fast.py`）：每轮 verify logits 与"从零重算"金标准（prefill+全量单次前向）的偏差，fast 引擎 max 0.38–0.59 / mean 0.30–0.33，老引擎 0.34–0.63 / 0.28–0.30——**同级**，即该偏差是模型 chunked/recurrent 路径的固有数值噪底（HF 自己的 decode 用 recurrent、prefill 用 chunked，同样存在）。两种引擎互为等价实现。

**12 prompt × 256 token 三方实测**（`test/dspark_fast_speedup.ipynb`，已执行，v2-B drafter）：

| 条件 | baseline | 老引擎 | **快速引擎** | 老 vs base | **fast vs base** | τ_old | τ_fast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T=1.0（论文口径） | 57.2 tok/s | 56.5 | **62.7** | 1.01× | **1.10×** | 3.79 | 2.41 |
| T=0.7/0.8/20 | 56.9 | 59.0 | **72.8** | 0.96× | **1.28×** | 3.92 | 2.80 |

每轮 37.7 ms = draft 6.6 + verify 30.1 + rewind 1.1；盈亏平衡 τ = 2.16（老引擎 3.9）。

**τ_fast ≠ τ_old 的结构性原因（重要）**：老引擎每轮提交 [anchor, k 草稿, corr] = k+2 个 token（2 次 target 前向里产出 2 个 target 采样 token）；快速引擎每轮提交 [corr(上轮), k 草稿] = k+1 个（1 次前向产出 1 个 target 采样 token）。**每轮少 1 个 token 但少 1 次前向**——单次前向的产出比 old 高一倍（(1+k)/1 vs (2+k)/2），tok/s 反超。τ 数字不能跨引擎直接比，只能比 tok/s。

**剩余瓶颈与下一步**：verify 前向 30.1 ms 中 GPU 只忙 ~5.8 ms（1798 次 kernel 发射）。torch.compile 实测只能 29.7→26.4 ms（fla Triton 核强制 graph break；集成进完整生成路径还会报错，已放弃，见 `test/try_compile.py`）。真正的下一档是**静态 KV cache + CUDA graph**（vLLM 式），预估 verify → ~8-10 ms、每轮 ~16 ms、T=1.0 加速比 → ~2.4×；其次是继续抬 τ（数据扩量）。

### 9.4 多分支 rollout 验证（2026-11-16 实现，用户提出并验证核心假设）

**思路**：每轮不再只采样 1 条 draft 链，而是采样 **B 条独立链**（共享 drafter base logits，仅 markov 采样不同；Gumbel 噪声各自独立），作为 **batch=B 的 B 个序列在同一个 target 前向里一起验证**，提交接受最深的链（+其修正 token）。这不是 tree attention——GDN 层的循环状态无法用 attention mask 分支隔离，分支在 **batch 维隔离**（各自复制前缀状态，GDN 快照 0.2ms×B + KV repeat）。

**关键实测**（本引擎是 CPU 发射瓶颈，GPU 忙仅 ~6ms）：批量验证前向墙钟 **B=1/2/4/8 全部 ~30ms**——验证 B 倍 token 墙钟免费。论文对树形验证的批评（验证 token 膨胀伤吞吐）适用于 GPU 吞吐受限的批量服务场景，不适用于单请求发射受限的场景。

**实现**（`dspark_fast.py`，`generate(..., branches=B)`）：per-slot 采样/接受判定向量化到 (B, n)；`_accept_multi` 取最深接受分支；拒绝时数学回滚用该分支的 hidden_states（`rewind(..., bidx)`）。正确性校验全部通过（Test A/B 与之前逐位同级）。

**12 prompt × 256 token 实测**（T=1.0，v2-B drafter，baseline 57.2 tok/s）：

| branches | τ_real | tok/s | vs baseline |
| --- | --- | --- | --- |
| 1 | 2.43 | 62.5 | 1.09× |
| 4 | 3.64 | **93.2** | **1.63×** |
| 8 | 4.11 | **101.2** | **1.77×** |

分支收益与 E[best-of-B] 理论吻合（按 α 向量预测 B=4 τ≈3.85）。域分化仍在：代码类 prompt B=4 达 113.9 tok/s（2.0×），低接受率闲聊类收益小（α 天花板限制）。B 的墙钟代价：状态复制+KV repeat 约 1-3ms，B=8 后边际收益递减；下一个乘法杠杆仍是 verify 前向的发射开销（CUDA graph + 静态 KV）与 drafter 域覆盖。

### 9.5 分层逆 CDF 采样（2026-11-16，用户设计）

多分支采样的两种模式（`generate(..., mode=)`）：**gumbel**（各分支 i.i.d. Gumbel-max，会重复抽中同一高概率块）与 **stratified**（默认）：整条 7-token 路径由**一个 uniform 经嵌套链式逆 CDF** 决定（u 落在块分布 CDF 的哪个嵌套区间就得到哪条路径），分支间用分层 uniform `u_b=(b−1+v_b)/B` 把 [0,1) 等分——**K 条路径按概率比例铺满块分布**。关键性质：u_b 边际均匀 → 每条路径仍是 P(block) 的精确采样 → 无损性保持（强制去重的无放回采样会破坏这一点，不可用）。代价：每槽 cumsum+searchsorted 比 argmax 多 ~2ms/轮。

12 prompt 实测（T=1.0，baseline 57.2）：B=4 stratified 87.0 tok/s（τ 3.72）vs gumbel 92.1（τ 3.64）；**B=8 stratified 105.6 tok/s（τ 4.44）= 1.85×**（gumbel B=8 101.2/4.11）。分层在高 B 下覆盖优势超过其额外开销，是当前最优配置。

### 9.6 设计文档

多分支引擎的完整符号表、推导与代码对照已单独成文：`doc/DSpark-多分支引擎.md`（含块分布的 [0,1) 表示、嵌套逆 CDF、分层 uniform 的无损性论证、GDN 数学回滚、成本分解与实测汇总）。
