#!/bin/bash
# Domain-expansion pipeline: wait for generation -> verify -> probe -> train (mix PB+domain,
# init from v2-B) -> eval BEFORE (v2-B) on GPU1 -> wait for training -> eval AFTER.
W=/home/dev/KAP/data/domain_exp
PB=/home/dev/KAP/data/pb_dspark
D=/home/dev/KAP/train
PAR=/home/dev/KAP/parameter
PY=/home/dev/venv/bin/python
CK=$PAR/draft/dspark_pb_v2_b/best.pt
OUT=$PAR/draft/dspark_domain
LOG=$W/pipeline.log
echo "$(date) domain pipeline started" >> $LOG

# 1) wait for generation workers
while pgrep -f '[p]b_gen_worker.py' > /dev/null; do sleep 30; done
echo "$(date) generation exited" >> $LOG

# 2) completeness
$PY -I - "$W" <<'EOF' >> $LOG 2>&1 || { echo "gen incomplete, abort" >> $LOG; exit 1; }
import glob, sys, torch
W = sys.argv[1]
items = torch.load(W + '/prompts.pt')
ids = {c['id'] for c in items}
got = set()
for f in sorted(glob.glob(W + '/gen/shard_r*_*.pt')):
    got.update(r['id'] for r in torch.load(f))
print(f'domain gen: {len(got)}/{len(ids)}')
assert got >= ids
EOF
echo "$(date) gen complete, probing" >> $LOG

# 3) probe
CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train_v2.py --anchors 32 --seqs 16 --epochs 2 \
  --data2 $W --init $CK --out $OUT --probe >> $W/probe.log 2>&1
tail -n 1 $W/probe.log >> $LOG

# 4) train on GPU0
nohup env CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train_v2.py --anchors 32 --seqs 16 --epochs 2 \
  --lr 3e-4 --data2 $W --init $CK --out $OUT > $W/train.log 2>&1 &
echo "$(date) training launched (GPU0)" >> $LOG

# 5) BEFORE eval on GPU1
CUDA_VISIBLE_DEVICES=1 $PY -I $D/../test/test_domain_eval.py --ckpt $CK --out $W/eval_before.json >> $W/eval.log 2>&1
echo "$(date) BEFORE eval done" >> $LOG

# 6) wait for training, then AFTER eval
while pgrep -f '[p]b_train_v2.py' > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 $PY -I $D/../test/test_domain_eval.py --ckpt $OUT/best.pt --out $W/eval_after.json >> $W/eval.log 2>&1
echo "$(date) AFTER eval done" >> $LOG
tail -n 3 $W/train.log >> $LOG
