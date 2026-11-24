#!/bin/bash
# Wait for the 0.7/0.8/20 generation (gen07) to finish, verify completeness,
# then launch the two v2 training runs (official DeepSpec recipe).
W=/home/dev/KAP/data/pb_dspark
D=/home/dev/KAP/train
P=/home/dev/KAP/parameter
PY=/home/dev/venv/bin/python
while pgrep -f '[p]b_gen_worker.py' > /dev/null; do sleep 30; done
echo "$(date) gen07 workers exited" >> $W/pipeline_v2.log

# completeness: every prompt id must have a response in gen07
$PY -I - "$W" <<'EOF' >> $W/pipeline_v2.log 2>&1 || exit 1
import glob, sys, torch
W = sys.argv[1]
items = torch.load(W + '/prompts.pt')
ids = {c['id'] for c in items}
got = set()
for f in sorted(glob.glob(W + '/gen07/shard_r*_*.pt')):
    got.update(r['id'] for r in torch.load(f))
missing = len(ids - got)
print(f'gen07: {len(got)}/{len(ids)} responses, missing {missing}')
assert missing == 0
EOF
echo "$(date) gen07 complete, probing v2 training" >> $W/pipeline_v2.log

# memory/time probe on the real data (3 steps each, then exit)
CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train_v2.py --anchors 32 --seqs 16 --probe \
  --out $P/draft/dspark_pb_v2_a >> $W/probe_a.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY -I $D/pb_train_v2.py --anchors 64 --seqs 8 --probe \
  --out $P/draft/dspark_pb_v2_b >> $W/probe_b.log 2>&1
echo "$(date) probes done, launching training runs" >> $W/pipeline_v2.log

nohup env CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train_v2.py --anchors 32 --seqs 16 --epochs 10 \
  --out $P/draft/dspark_pb_v2_a > $W/train_v2_a.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 $PY -I $D/pb_train_v2.py --anchors 64 --seqs 8 --epochs 10 \
  --out $P/draft/dspark_pb_v2_b > $W/train_v2_b.log 2>&1 &
echo "$(date) training v2 launched (a: 32x16, b: 64x8)" >> $W/pipeline_v2.log
