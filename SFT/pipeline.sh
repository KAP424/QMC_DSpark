#!/bin/bash
# Full pipeline: SFT (wave1: maze+sudoku, wave2: countdown+arc) →
#               drafter gen → drafter train, all using /home/dev/KAP/parameter/
#
# nohup bash /home/dev/KAP/SFT/pipeline.sh > /home/dev/KAP/SFT/pipeline.log 2>&1 &

PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter
ORIG=$PAR/target/Qwen3.5-0.8B-Base-original
TRAIN=/home/dev/KAP/train
DATA=/home/dev/KAP/data
LOG=/home/dev/KAP/SFT/pipeline.log

echo "$(date) pipeline started" | tee -a $LOG

# ── SFT wave 1: maze (GPU0) + sudoku (GPU1) in parallel ──────────────────────
echo "$(date) wave1: maze SFT GPU0 + sudoku SFT GPU1" | tee -a $LOG

CUDA_VISIBLE_DEVICES=0 $PY -I /home/dev/KAP/SFT/sft_train.py \
    --dataset maze \
    --out $PAR/target/Qwen3.5-0.8B-Base-maze-sft \
    --bs 8 --grad-accum 4 --epochs 1 --lr 1e-5 \
    > $PAR/target/Qwen3.5-0.8B-Base-maze-sft/train.log 2>&1 &
MAZE_SFT_PID=$!

CUDA_VISIBLE_DEVICES=1 $PY -I /home/dev/KAP/SFT/sft_train.py \
    --dataset sudoku \
    --out $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft \
    --bs 8 --grad-accum 4 --epochs 1 --lr 1e-5 \
    > $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/train.log 2>&1 &
SUDOKU_SFT_PID=$!

wait $MAZE_SFT_PID $SUDOKU_SFT_PID
echo "$(date) wave1 done" | tee -a $LOG

# ── SFT wave 2: countdown (GPU0) + arc (GPU1) in parallel ────────────────────
echo "$(date) wave2: countdown SFT GPU0 + arc SFT GPU1" | tee -a $LOG

CUDA_VISIBLE_DEVICES=0 $PY -I /home/dev/KAP/SFT/sft_train.py \
    --dataset countdown \
    --out $PAR/target/Qwen3.5-0.8B-Base-countdown-sft \
    --bs 8 --grad-accum 4 --epochs 1 --lr 1e-5 \
    > $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/train.log 2>&1 &
CD_SFT_PID=$!

CUDA_VISIBLE_DEVICES=1 $PY -I /home/dev/KAP/SFT/sft_train.py \
    --dataset 1d_arc \
    --out $PAR/target/Qwen3.5-0.8B-Base-arc-sft \
    --bs 8 --grad-accum 4 --epochs 1 --lr 1e-5 \
    > $PAR/target/Qwen3.5-0.8B-Base-arc-sft/train.log 2>&1 &
ARC_SFT_PID=$!

wait $CD_SFT_PID $ARC_SFT_PID
echo "$(date) wave2 done" | tee -a $LOG

# ── Drafter gen function ──────────────────────────────────────────────────────
# $1=domain $2=GPU $3=data_dir $4=gen_subdir $5=sft_model_dir
gen_drafter_data() {
    local DS=$1 GPU=$2 D=$3 GEN=$4 TARGET=$5
    echo "$(date) [$DS] gen drafter data on GPU$GPU" | tee -a $LOG
    mkdir -p $D/$GEN
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $TRAIN/pb_gen_worker.py \
        --rank 0 --world 1 \
        --work $D \
        --target $TARGET \
        --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
        --out $GEN \
        > $D/gen_${DS}.log 2>&1
    echo "$(date) [$DS] gen done" | tee -a $LOG
}

# ── Drafter train function ────────────────────────────────────────────────────
# $1=domain $2=GPU $3=data_dir $4=gen_subdir $5=draft_out_dir
train_drafter() {
    local DS=$1 GPU=$2 D=$3 GEN=$4 OUT=$5
    local INIT=$PAR/draft/dspark_maze_sudoku/best.pt
    echo "$(date) [$DS] training drafter on GPU$GPU -> $OUT" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $TRAIN/pb_train_v2.py \
        --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
        --data $D --gen-dir $GEN \
        --init $INIT \
        --out $OUT \
        > $OUT/train.log 2>&1
    echo "$(date) [$DS] drafter done, best tau=$(python3 -I -c \
        "import torch; c=torch.load('$OUT/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)" \
        | tee -a $LOG
}

# ── Drafter wave 1: maze (GPU0) + sudoku (GPU1) ───────────────────────────────
echo "$(date) drafter wave1: maze+sudoku gen" | tee -a $LOG

gen_drafter_data maze   0 $DATA/maze_sudoku gen_sft_maze   $PAR/target/Qwen3.5-0.8B-Base-maze-sft/hf_model &
gen_drafter_data sudoku 1 $DATA/maze_sudoku gen_sft_sudoku $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model &
wait
echo "$(date) maze+sudoku gen done, starting drafter train" | tee -a $LOG

# maze and sudoku share the same prompts.pt so train needs separate data dirs
# maze: use maze_sudoku dir with maze-only prompts subset via gen filter
# For now train on pb_dspark (domain-agnostic base) + domain gen data
# maze drafter
train_drafter maze   0 $DATA/maze_sudoku   gen_sft_maze   $PAR/draft/dspark_maze_sft &
train_drafter sudoku 1 $DATA/maze_sudoku   gen_sft_sudoku $PAR/draft/dspark_sudoku_sft &
wait
echo "$(date) maze+sudoku drafter wave1 done" | tee -a $LOG

# ── Drafter wave 2: countdown (GPU0) + arc (GPU1) ────────────────────────────
echo "$(date) drafter wave2: countdown+arc gen" | tee -a $LOG

gen_drafter_data countdown 0 $DATA/domain_exp gen_sft_cd  $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model &
gen_drafter_data 1d_arc    1 $DATA/domain_exp gen_sft_arc $PAR/target/Qwen3.5-0.8B-Base-arc-sft/hf_model &
wait
echo "$(date) countdown+arc gen done, starting drafter train" | tee -a $LOG

train_drafter countdown 0 $DATA/domain_exp gen_sft_cd  $PAR/draft/dspark_countdown_sft &
train_drafter 1d_arc    1 $DATA/domain_exp gen_sft_arc $PAR/draft/dspark_arc_sft &
wait

echo "$(date) FULL PIPELINE COMPLETE" | tee -a $LOG
echo "" | tee -a $LOG
echo "Target models:" | tee -a $LOG
for d in maze sudoku countdown arc; do
    f=$PAR/target/Qwen3.5-0.8B-Base-${d}-sft/train.log
    tail -1 $f 2>/dev/null | tee -a $LOG
done
echo "Draft models:" | tee -a $LOG
for d in maze sudoku countdown arc; do
    f=$PAR/draft/dspark_${d}_sft/train.log
    tail -1 $f 2>/dev/null | tee -a $LOG
done
