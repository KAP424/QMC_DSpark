#!/bin/bash
# Retrain all 4 domain drafters against their own SFT target.
# Run entirely on GPU 1 (GPU 0 may be busy).
# Each domain: gen T=0.7 responses → gen T=1.0 val responses → train drafter.
set -e

PY=/home/dev/venv/bin/python
D=/home/dev/KAP/train
PAR=/home/dev/KAP/parameter
MS=/home/dev/KAP/data/maze_sudoku
DE=/home/dev/KAP/data/domain_exp
LOG_DIR=/home/dev/KAP/data/sft_retrain_logs
mkdir -p $LOG_DIR

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU

run_domain() {
    local DS=$1          # maze | sudoku | countdown | 1d_arc
    local WORK=$2        # data root containing prompts.pt
    local TARGET=$3      # SFT target hf_model path
    local OUT=$4         # output draft checkpoint dir
    local MAX_NEW=$5     # max new tokens for gen
    local LOG=$LOG_DIR/${DS}.log

    echo "$(date)  === $DS ===" | tee -a $LOG

    # 1) Generate T=0.7 training responses (official recipe)
    echo "$(date)  $DS: generating train responses (T=0.7)" | tee -a $LOG
    $PY -I $D/pb_gen_worker.py \
        --rank 0 --world 1 \
        --work $WORK \
        --target $TARGET \
        --ds-filter $DS \
        --temperature 0.7 --top-p 0.8 --top-k 20 \
        --max-new $MAX_NEW \
        --out gen_sft \
        >> $LOG 2>&1
    echo "$(date)  $DS: train gen done" | tee -a $LOG

    # 2) Generate T=1.0 val responses (paper-protocol eval) — all items so val-split IDs are covered
    echo "$(date)  $DS: generating val responses (T=1.0)" | tee -a $LOG
    $PY -I $D/pb_gen_worker.py \
        --rank 0 --world 1 \
        --work $WORK \
        --target $TARGET \
        --ds-filter $DS \
        --temperature 1.0 \
        --max-new $MAX_NEW \
        --out gen_sft_val \
        >> $LOG 2>&1
    echo "$(date)  $DS: val gen done" | tee -a $LOG

    # 3) Train drafter against SFT target
    echo "$(date)  $DS: training drafter" | tee -a $LOG
    $PY -I $D/pb_train_v2.py \
        --data $WORK \
        --gen-dir gen_sft \
        --val-gen-dir gen_sft_val \
        --target $TARGET \
        --ds-filter $DS \
        --anchors 32 --seqs 16 --epochs 10 --lr 6e-4 \
        --out $OUT \
        >> $LOG 2>&1
    echo "$(date)  $DS: training done" | tee -a $LOG
}

# Run all 4 domains sequentially
run_domain maze      $MS $PAR/target/Qwen3.5-0.8B-Base-maze-sft/hf_model      $PAR/draft/dspark_maze_sft      256
run_domain sudoku    $MS $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model    $PAR/draft/dspark_sudoku_sft    512
run_domain countdown $DE $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model $PAR/draft/dspark_countdown_sft 256
run_domain 1d_arc    $DE $PAR/target/Qwen3.5-0.8B-Base-arc-sft/hf_model       $PAR/draft/dspark_arc_sft       256

echo "$(date)  all 4 domains done" | tee -a $LOG_DIR/all.log
