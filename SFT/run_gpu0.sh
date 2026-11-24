#!/bin/bash
# GPU0: maze → countdown (fully independent, never touches GPU1)
PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter
TRAIN=/home/dev/KAP/train
DATA=/home/dev/KAP/data
LOG=/home/dev/KAP/SFT/gpu0.log
INIT=$PAR/draft/dspark_maze_sudoku/best.pt

echo "$(date) GPU0 pipeline started" | tee -a $LOG

# ── maze ──────────────────────────────────────────────────────────────────────
# gen (maze shards already done: 79 shards present — skip if exists)
if [ $(ls $DATA/maze_sudoku/gen_sft_maze/*.pt 2>/dev/null | wc -l) -ge 10 ]; then
    echo "$(date) [maze] gen already done, skipping" | tee -a $LOG
else
    echo "$(date) [maze] gen" | tee -a $LOG
    mkdir -p $DATA/maze_sudoku/gen_sft_maze
    CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_gen_worker.py \
        --rank 0 --world 1 --work $DATA/maze_sudoku \
        --target $PAR/target/Qwen3.5-0.8B-Base-maze-sft/hf_model \
        --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
        --out gen_sft_maze >> $DATA/maze_sudoku/gen_maze.log 2>&1
    echo "$(date) [maze] gen done" | tee -a $LOG
fi

echo "$(date) [maze] drafter train" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/maze_sudoku --gen-dir gen_sft_maze \
    --init $INIT --out $PAR/draft/dspark_maze_sft \
    > $PAR/draft/dspark_maze_sft/train.log 2>&1
TAU=$($PY -I -c "import torch; c=torch.load('$PAR/draft/dspark_maze_sft/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)
echo "$(date) [maze] drafter done  tau=$TAU" | tee -a $LOG

# ── countdown ─────────────────────────────────────────────────────────────────
echo "$(date) [countdown] gen" | tee -a $LOG
mkdir -p $DATA/domain_exp/gen_sft_cd
CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_gen_worker.py \
    --rank 0 --world 1 --work $DATA/domain_exp \
    --target $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model \
    --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
    --out gen_sft_cd >> $DATA/domain_exp/gen_countdown.log 2>&1
echo "$(date) [countdown] gen done" | tee -a $LOG

echo "$(date) [countdown] drafter train" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/domain_exp --gen-dir gen_sft_cd \
    --init $INIT --out $PAR/draft/dspark_countdown_sft \
    > $PAR/draft/dspark_countdown_sft/train.log 2>&1
TAU=$($PY -I -c "import torch; c=torch.load('$PAR/draft/dspark_countdown_sft/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)
echo "$(date) [countdown] drafter done  tau=$TAU" | tee -a $LOG

echo "$(date) GPU0 pipeline COMPLETE" | tee -a $LOG
