# src4dspark：DSpark 模型与推理源码

| 文件 | 内容 |
| --- | --- |
| `dspark_paper.py` | **现行 drafter** `DSparkDrafter`：5 层，整条前缀的 5 层 target 特征注入每层 K/V（上下文与草稿行共享 W_K/W_V），块内双向，Markov 头 rank 128，置信度头。`ND = 7`（γ）在这里定义。 |
| `dflash_big.py` | 基础模块：`RMSNorm`、`MarkovHead`、`CTX_LAYERS`/`CTX_HS`（取哪几层 target 特征）、`save`/`load`；另含旧版 4 层 drafter `DFlashDrafterBig`。`dspark_paper.py` 依赖它。 |
| `dspark_e2e.py` | 端到端解码：`DSparkDecode`（anchor 采样 → 7 个草稿 → target 一次验证 → 拒绝采样；部分接受时 deepcopy 回滚并重放）和对照用的 `baseline_generate`。 |
| `legacy/` | 早期实验归档（MTP drafter、DFlash v1/v2、rollout、分层采样、K 扫描等，以及它们的训练脚本）。这些文件互相 import，所以整体放在一起。路径已改到新位置，只做过编译和 import 检查，**没有重新运行**。 |

使用：

```python
import sys
sys.path.insert(0, '/home/dev/KAP/src4dspark')
from dspark_paper import DSparkDrafter, load, ND
from dspark_e2e import DSparkDecode, baseline_generate

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
# model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map='cuda:0').eval()
# d = DSparkDrafter(model, device='cuda:0', dtype=torch.float32)
# load(d, '/home/dev/KAP/parameter/draft/dspark_pb/dflash_pb_best.pt')
```

每 token 耗时对比的 notebook 在 `/home/dev/KAP/test/dspark_latency.ipynb`。
