#!/bin/bash
# GPU1: sudoku → arc (fully independent, never touches GPU0)
PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter
TRAIN=/home/dev/KAP/train
DATA=/home/dev/KAP/data
LOG=/home/dev/KAP/SFT/gpu1.log
INIT=$PAR/draft/dspark_maze_sudoku/best.pt

echo "$(date) GPU1 pipeline started" | tee -a $LOG

# ── sudoku ────────────────────────────────────────────────────────────────────
echo "$(date) [sudoku] gen" | tee -a $LOG
mkdir -p $DATA/maze_sudoku/gen_sft_sudoku
CUDA_VISIBLE_DEVICES=1 /home/dev/venv/bin/python -I $TRAIN/pb_gen_worker.py \
    --rank 0 --world 1 --work $DATA/maze_sudoku \
    --target $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model \
    --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
    --out gen_sft_sudoku >> $DATA/maze_sudoku/gen_sudoku.log 2>&1
echo "$(date) [sudoku] gen done" | tee -a $LOG

echo "$(date) [sudoku] drafter train" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/maze_sudoku --gen-dir gen_sft_sudoku \
    --init $INIT --out $PAR/draft/dspark_sudoku_sft \
    > $PAR/draft/dspark_sudoku_sft/train.log 2>&1
TAU=$($PY -I -c "import torch; c=torch.load('$PAR/draft/dspark_sudoku_sft/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)
echo "$(date) [sudoku] drafter done  tau=$TAU" | tee -a $LOG

# ── arc ───────────────────────────────────────────────────────────────────────
echo "$(date) [arc] gen" | tee -a $LOG
mkdir -p $DATA/domain_exp/gen_sft_arc
CUDA_VISIBLE_DEVICES=1 $PY -I $TRAIN/pb_gen_worker.py \
    --rank 0 --world 1 --work $DATA/domain_exp \
    --target $PAR/target/Qwen3.5-0.8B-Base-arc-sft/hf_model \
    --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
    --out gen_sft_arc >> $DATA/domain_exp/gen_arc.log 2>&1
echo "$(date) [arc] gen done" | tee -a $LOG

echo "$(date) [arc] drafter train" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/domain_exp --gen-dir gen_sft_arc \
    --init $INIT --out $PAR/draft/dspark_arc_sft \
    > $PAR/draft/dspark_arc_sft/train.log 2>&1
TAU=$($PY -I -c "import torch; c=torch.load('$PAR/draft/dspark_arc_sft/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)
echo "$(date) [arc] drafter done  tau=$TAU" | tee -a $LOG

echo "$(date) GPU1 pipeline COMPLETE" | tee -a $LOG
