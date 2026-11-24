#!/bin/bash
# Re-run drafter gen + train for all 4 domains using the already-trained SFT targets.
# nohup bash /home/dev/KAP/SFT/drafter_pipeline.sh > /home/dev/KAP/SFT/drafter_pipeline.log 2>&1 &

PY=/home/dev/venv/bin/python
PAR=/home/dev/KAP/parameter
TRAIN=/home/dev/KAP/train
DATA=/home/dev/KAP/data
LOG=/home/dev/KAP/SFT/drafter_pipeline.log

echo "$(date) drafter pipeline started" | tee -a $LOG

gen_drafter_data() {
    local DS=$1 GPU=$2 D=$3 GEN=$4 TARGET=$5
    echo "$(date) [$DS] gen on GPU$GPU target=$TARGET" | tee -a $LOG
    mkdir -p $D/$GEN
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $TRAIN/pb_gen_worker.py \
        --rank 0 --world 1 \
        --work $D \
        --target $TARGET \
        --max-new 512 --temperature 0.7 --top-p 0.8 --top-k 20 --bs 64 \
        --out $GEN \
        > $D/gen_${DS}.log 2>&1
    echo "$(date) [$DS] gen done  ($(ls $D/$GEN/*.pt 2>/dev/null | wc -l) shards)" | tee -a $LOG
}

train_drafter() {
    local DS=$1 GPU=$2 D=$3 GEN=$4 OUT=$5
    local INIT=$PAR/draft/dspark_maze_sudoku/best.pt
    echo "$(date) [$DS] drafter train on GPU$GPU -> $OUT" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $TRAIN/pb_train_v2.py \
        --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
        --data $D --gen-dir $GEN \
        --init $INIT \
        --out $OUT \
        > $OUT/train.log 2>&1
    TAU=$($PY -I -c "import torch; c=torch.load('$OUT/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null)
    echo "$(date) [$DS] drafter done  tau=$TAU" | tee -a $LOG
}

# Wave 1: maze (GPU0) + sudoku (GPU1) gen in parallel
gen_drafter_data maze   0 $DATA/maze_sudoku gen_sft_maze \
    $PAR/target/Qwen3.5-0.8B-Base-maze-sft/hf_model &
gen_drafter_data sudoku 1 $DATA/maze_sudoku gen_sft_sudoku \
    $PAR/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model &
wait
echo "$(date) wave1 gen done" | tee -a $LOG

# Wave 1: drafter train in parallel
train_drafter maze   0 $DATA/maze_sudoku gen_sft_maze   $PAR/draft/dspark_maze_sft &
train_drafter sudoku 1 $DATA/maze_sudoku gen_sft_sudoku $PAR/draft/dspark_sudoku_sft &
wait
echo "$(date) wave1 drafter train done" | tee -a $LOG

# Wave 2: countdown (GPU0) + arc (GPU1) gen in parallel
gen_drafter_data countdown 0 $DATA/domain_exp gen_sft_cd \
    $PAR/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model &
gen_drafter_data 1d_arc    1 $DATA/domain_exp gen_sft_arc \
    $PAR/target/Qwen3.5-0.8B-Base-arc-sft/hf_model &
wait
echo "$(date) wave2 gen done" | tee -a $LOG

# Wave 2: drafter train in parallel
train_drafter countdown 0 $DATA/domain_exp gen_sft_cd  $PAR/draft/dspark_countdown_sft &
train_drafter 1d_arc    1 $DATA/domain_exp gen_sft_arc $PAR/draft/dspark_arc_sft &
wait
echo "$(date) wave2 drafter train done" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== RESULTS ===" | tee -a $LOG
for ds in maze sudoku countdown 1d_arc; do
    name=$(echo $ds | sed 's/1d_arc/arc/')
    OUT=$PAR/draft/dspark_${name}_sft
    TAU=$($PY -I -c "import torch; c=torch.load('$OUT/best.pt',map_location='cpu',weights_only=False); print(f'{c.get(\"tau\",\"?\"):.3f}')" 2>/dev/null || echo "?")
    echo "  dspark_${name}_sft: tau=$TAU" | tee -a $LOG
done
echo "$(date) DONE" | tee -a $LOG
