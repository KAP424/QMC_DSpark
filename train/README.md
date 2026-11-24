# train：训练 DSpark drafter 的源码

依赖 `/home/dev/KAP/src4dspark`（脚本会自动把它加入 `sys.path`）。target 默认用只读的原始拷贝 `/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original`，全程冻结。

| 文件 | 作用 |
| --- | --- |
| `pb_gen_worker.py` | 每张 GPU 一个进程，用 target 以纯温度采样为 prompt 生成 response，分片落盘，可断点续跑 |
| `pb_check_gen.py` | 检查 50,500 条 response 是否齐全 |
| `pb_after_gen.sh` | 等生成进程退出 → 校验数据 → 启动训练 |
| `pb_train.py` | 后台训练；指标写入 `data/pb_dspark/train_log.jsonl`；权重和续训状态写入 `parameter/draft/dspark_pb/` |
| `dspark_loss.py` | 取 block、损失（0.1·CE + 0.9·TV + 1.0·BCE，w_k = exp(−(k−1)/7)）、评测 |
| `sts_calibrate.py` | 置信度头的 STS 校准（实测无收益，未启用） |
| `dspark_plot.py` | 读训练日志画曲线 |
| `dspark_pb_live.ipynb` | 实时训练监控 |
| `dspark_pb_train.ipynb` | 一体化 notebook 版流程（未运行；实际训练用的是上面的脚本） |

数据：`/home/dev/KAP/data/pb_dspark/`（prompt、生成的 response、训练日志）。

续训（例如再训到 15 个 epoch）：

```bash
cd /home/dev
CUDA_VISIBLE_DEVICES=0 /home/dev/venv/bin/python -I /home/dev/KAP/train/pb_train.py --epochs 15 --temp 1.0
```

会从 `parameter/draft/dspark_pb/train_state.pt` 接着跑；总步数变化时调度器按新长度重建并快进。

详细说明见 `/home/dev/KAP/doc/DSpark-训练文档.md`。
