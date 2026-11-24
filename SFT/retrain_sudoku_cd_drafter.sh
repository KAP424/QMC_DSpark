#!/bin/bash
PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter
TRAIN=/home/dev/KAP/train
DATA=/home/dev/KAP/data
LOG=/home/dev/KAP/SFT/retrain2.log
INIT=$PAR/draft/dspark_maze_sudoku/best.pt

echo "$(date) retrain sudoku+countdown drafter started" | tee -a $LOG

# sudoku gen (GPU0) + countdown gen (GPU1) in parallel
mkdir -p $DATA/maze_sudoku/gen_sft_sudoku2 $DATA/domain_exp/gen_sft_cd2

CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_gen_worker.py \
    --rank 0 --world 1 --work $DATA/maze_sudoku \
    --target $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model \
    --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
    --out gen_sft_sudoku2 > $DATA/maze_sudoku/gen_sudoku2.log 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 $PY -I $TRAIN/pb_gen_worker.py \
    --rank 0 --world 1 --work $DATA/domain_exp \
    --target $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model \
    --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
    --out gen_sft_cd2 > $DATA/domain_exp/gen_cd2.log 2>&1 &
P1=$!

wait $P0
echo "$(date) sudoku gen done" | tee -a $LOG
wait $P1
echo "$(date) countdown gen done" | tee -a $LOG

# drafter train in parallel
CUDA_VISIBLE_DEVICES=0 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/maze_sudoku --gen-dir gen_sft_sudoku2 \
    --init $INIT --out $PAR/draft/dspark_sudoku_sft \
    > $PAR/draft/dspark_sudoku_sft/train2.log 2>&1 &
P2=$!

CUDA_VISIBLE_DEVICES=1 $PY -I $TRAIN/pb_train_v2.py \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data $DATA/domain_exp --gen-dir gen_sft_cd2 \
    --init $INIT --out $PAR/draft/dspark_countdown_sft \
    > $PAR/draft/dspark_countdown_sft/train2.log 2>&1 &
P3=$!

wait $P2
echo "$(date) sudoku drafter done" | tee -a $LOG
wait $P3
echo "$(date) countdown drafter done" | tee -a $LOG

for d in sudoku countdown; do
    TAU=$($PY -I -c "import torch; c=torch.load('$PAR/draft/dspark_${d}_sft/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",0):.3f}')" 2>/dev/null)
    echo "  dspark_${d}_sft  tau=$TAU" | tee -a $LOG
done
echo "$(date) ALL DONE" | tee -a $LOG
