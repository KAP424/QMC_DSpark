# DSpark 推断策略配置与实验记录 (2026-11-24)

本文件记录了在 Maze (100题) 与 Sudoku 任务上不同采样策略的对比。

## 1. 采样设置定义 (`SETTINGS`)
所有设置均基于 DSpark 解码引擎。

| 设置 ID | 架构类型 | Draft 策略 | Accept 策略 | Bonus 策略 |
| :--- | :--- | :--- | :--- | :--- |
| **S1_iid** | Classic | i.i.d. | i.i.d. | i.i.d. |
| **S2_draft** | Classic | Lattice | i.i.d. | i.i.d. |
| **S3_draft_acc** | Classic | Lattice | Lattice | i.i.d. |
| **S4_draft_acc_bns** | Classic | Lattice | Lattice | Lattice |
| **S5_dspark_ac** | AC | 确定性映射 | 确定性映射 | 确定性映射 |
| **S6_paper** | Paper AC | 无 drafter | 无 drafter | 无 drafter |

### 核心术语定义：
*   **Classic (传统投机采样)**: 依赖经典拒绝采样规则 $\min(1, p/q)$。
*   **AC (算术编码确定性解码)**: 序列生成被建模为目标分布 $p$ 下的算术解码过程 $\Phi_p(u_b)$。全过程由单一随机数 $u_b$ 驱动。
*   **Lattice (格耦合)**: 通过将 $u$ 进行 $u_b = (\Delta + b/64) \bmod 1$ 的重采样，强制 $B$ 个分支在概率空间中均匀分布。
EOF
