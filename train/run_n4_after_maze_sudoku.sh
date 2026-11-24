#!/bin/bash
# Wait for maze_sudoku pipeline to finish, then run n=4 drafter experiment.
# Launch with: nohup bash /home/dev/KAP/train/run_n4_after_maze_sudoku.sh &

PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter/draft
OUT=$PAR/dspark_n4
LOG=/home/dev/KAP/data/maze_sudoku/n4_run.log
MAZE_DATA=/home/dev/KAP/data/maze_sudoku
PB_DATA=/home/dev/KAP/data/pb_dspark

mkdir -p $OUT
echo "$(date) n=4 watcher started" >> $LOG

# 1) wait for maze_sudoku pipeline (pb_maze_sudoku_pipeline.sh runs pb_train_v2 on GPU0)
while pgrep -f '[p]b_maze_sudoku_pipeline\|[p]b_train_v2' > /dev/null; do
    sleep 60
done
echo "$(date) maze_sudoku pipeline done; starting n=4 training" >> $LOG

# 2) train n=4 — init from maze_sudoku best (has maze+sudoku+PB knowledge)
#    GPU0 is free now; run epochs=2, same LR as maze_sudoku run
nohup env CUDA_VISIBLE_DEVICES=0 $PY -I /home/dev/KAP/train/pb_train_v2.py \
    --n-draft 4 \
    --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
    --data  $PB_DATA \
    --data2 $MAZE_DATA \
    --init  $PAR/dspark_maze_sudoku/best.pt \
    --out   $OUT \
    > $OUT/train.log 2>&1
echo "$(date) n=4 training done" >> $LOG

# 3) compare n=4 model (inference n=4) vs n=7 model (inference n=6) on PB val
env CUDA_VISIBLE_DEVICES=0 $PY -I /home/dev/KAP/test/test_n4_vs_n7.py \
    --ckpt4 $OUT/best.pt \
    --ckpt7 $PAR/dspark_maze_sudoku/best.pt \
    > $OUT/compare.log 2>&1
echo "$(date) comparison done; see $OUT/compare.log" >> $LOG
