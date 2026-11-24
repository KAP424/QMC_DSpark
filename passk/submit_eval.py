import argparse
import subprocess
import time
from pathlib import Path

# 定义任务配置 (任务名: (GPU_ID, 数据集, drafter, target, max_new, ub_mode))
TASKS = {
    'maze':      ('0', '/home/dev/KAP/data/maze_sudoku/prompts.pt', '/home/dev/KAP/parameter/draft/dspark_maze_sft/best.pt', '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-maze-sft/hf_model', 64, 'gt_no_eos'),
    'sudoku':    ('0', '/home/dev/KAP/data/maze_sudoku/prompts.pt', '/home/dev/KAP/parameter/draft/dspark_sudoku_sft/best.pt', '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model', 200, 'gt_no_eos'),
    'countdown': ('1', '/home/dev/KAP/data/domain_exp/prompts.pt', '/home/dev/KAP/parameter/draft/dspark_countdown_sft/best.pt', '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model', 64, 'mc'),
    '1d_arc':    ('1', '/home/dev/KAP/data/domain_exp/prompts.pt', '/home/dev/KAP/parameter/draft/dspark_arc_sft/best.pt', '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original', 200, 'gt_no_eos')
}

def run_task(task_name, smoke=True):
    gpu, data, drafter, target, max_new, ub = TASKS[task_name]
    cmd = [
        "/home/dev/venv/bin/python", "-I", "/home/dev/KAP/passk/run_passk.py",
        "--domain", task_name,
        "--device", f"cuda:{gpu}",
        "--n-prompts", "2" if smoke else "100" # smoke test 模式
    ]
    print(f"Running {'SMOKE TEST' if smoke else 'FULL EVAL'} for {task_name} on GPU {gpu}...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (2 prompts)")
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    tasks = TASKS.keys() if args.task == "all" else [args.task]
    for t in tasks:
        run_task(t, smoke=args.smoke)
