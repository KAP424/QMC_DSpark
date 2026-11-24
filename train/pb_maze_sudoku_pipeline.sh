#!/bin/bash
# Maze + Sudoku domain pipeline:
# wait for gen → verify → train (mix PB+maze_sudoku, init from domain ckpt) → eval
W=/home/dev/KAP/data/maze_sudoku
PB=/home/dev/KAP/data/pb_dspark
D=/home/dev/KAP/train
PAR=/home/dev/KAP/parameter
PY=/home/dev/venv/bin/python
INIT=$PAR/draft/dspark_domain/best.pt
OUT=$PAR/draft/dspark_maze_sudoku
LOG=$W/pipeline.log
mkdir -p $OUT
echo "$(date) maze_sudoku pipeline started" >> $LOG

# 1) wait for gen workers
while pgrep -f '[p]b_gen_worker.py' > /dev/null; do sleep 30; done
echo "$(date) generation done" >> $LOG

# 2) completeness check
$PY -I - "$W" <<'EOF' >> $LOG 2>&1 || { echo "gen incomplete, abort" >> $LOG; exit 1; }
import glob, sys, torch
W = sys.argv[1]
items = torch.load(W + '/prompts.pt')
ids = {c['id'] for c in items if c['split'] == 'train'}
got = set()
for f in sorted(glob.glob(W + '/gen/shard_r*_*.pt')):
    got.update(r['id'] for r in torch.load(f))
print(f'maze_sudoku gen: {len(got)}/{len(ids)}')
assert got >= ids
EOF
echo "$(date) gen complete" >> $LOG

# 3) train on GPU0 — init from dspark_domain, mix in PB
nohup env CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train_v2.py \
  --anchors 32 --seqs 16 --epochs 2 --lr 2e-4 \
  --data2 $W --init $INIT --out $OUT > $W/train.log 2>&1 &
echo "$(date) training launched (GPU0)" >> $LOG

# 4) BEFORE eval on GPU1 (domain ckpt)
CUDA_VISIBLE_DEVICES=1 $PY -I $D/../test/test_domain_eval.py \
  --ckpt $INIT --out $W/eval_before.json >> $W/eval.log 2>&1
echo "$(date) BEFORE eval done" >> $LOG

# 5) wait for training → AFTER eval
while pgrep -f '[p]b_train_v2.py' > /dev/null; do sleep 60; done
echo "$(date) training done" >> $LOG
CUDA_VISIBLE_DEVICES=1 $PY -I $D/../test/test_domain_eval.py \
  --ckpt $OUT/best.pt --out $W/eval_after.json >> $W/eval.log 2>&1
echo "$(date) AFTER eval done" >> $LOG
# append maze+sudoku val τ (branches=8) to compare
CUDA_VISIBLE_DEVICES=1 $PY -I - <<'PYEOF' >> $LOG 2>&1
import json, sys, time, torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
import dspark_fast
P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
W = '/home/dev/KAP/data/maze_sudoku'
PAR = '/home/dev/KAP/parameter/draft'
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map='cuda:0').eval()
for p in model.parameters(): p.requires_grad_(False)
items = torch.load(W + '/prompts.pt')
out = {}
for ckpt_name, ckpt in (('before', PAR+'/dspark_domain/best.pt'),
                         ('after',  PAR+'/dspark_maze_sudoku/best.pt')):
    d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device='cuda:0', dtype=torch.float32)
    c = torch.load(ckpt, map_location='cuda:0', weights_only=False)
    d.load_state_dict(c['model']); d.eval()
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=11)
    out[ckpt_name] = {}
    for ds in ('maze', 'sudoku'):
        ps = [it['ids'] for it in items if it['split']=='val' and it['dataset']==ds][:16]
        taus = []
        for pr in ps:
            _, st = eng.generate(pr, 256, branches=8, mode='stratified')
            taus.append(st['tokens']/st['rounds'])
        out[ckpt_name][ds] = {'tau': sum(taus)/len(taus)}
        print(f'{ckpt_name} {ds}: tau {sum(taus)/len(taus):.3f}')
    del d, eng
json.dump(out, open(W+'/tau_before_after.json','w'), indent=1)
PYEOF
echo "$(date) full eval done" >> $LOG
