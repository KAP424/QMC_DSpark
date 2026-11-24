'''Create the domain-expansion prompt set: countdown + 1d-ARC, rendered for the Base target.
Outputs /home/dev/KAP/data/domain_exp/prompts.pt with the same schema as pb_dspark/prompts.pt
(items: id, ids (token ids), split train/val). 200 val prompts held out (100 per domain).'''
import glob, json, os, random, sys

import torch
from transformers import AutoTokenizer

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
OUT = '/home/dev/KAP/data/domain_exp'
N_COUNTDOWN, N_ARC, N_VAL = 20_000, 20_000, 100
rng = random.Random(2026)

os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P)

items = []

# --- countdown prompts ---
import pyarrow.parquet as pq
t = pq.read_table('/home/dev/KAP/data/countdown/data/train-00000-of-00001.parquet')
idx = rng.sample(range(t.num_rows), N_COUNTDOWN + N_VAL)
for n, i in enumerate(idx):
    tgt = t.column('target')[i].as_py()
    nums = t.column('nums')[i].as_py()
    text = (f'Using the numbers {nums}, create an expression that equals {tgt}. '
            'You can use basic arithmetic operations (+, -, *, /). Each number must be used exactly once.')
    items.append({'dataset': 'countdown', 'ids': tok(text)['input_ids'],
                  'split': 'val' if n >= N_COUNTDOWN else 'train',
                  'raw': {'nums': nums, 'target': tgt}})

# --- 1d-ARC prompts ---
files = sorted(glob.glob('/home/dev/KAP/data/1d_arc/1D-ARC-main/dataset/*/*.json'))
pick = rng.choices(files, k=N_ARC + N_VAL)          # 901 个任务，有放回；多样性来自 target 的随机生成
for n, f in enumerate(pick):
    task = json.load(open(f))
    demos, test = task['train'], task['test'][0]
    row = lambda g: ' '.join(map(str, g[0]))
    parts = ['Apply the same transformation rule to the test input.',
             'Here are example input/output pairs:']
    for dmo in demos:
        parts += ['Input:', row(dmo['input']), 'Output:', row(dmo['output'])]
    parts += ['Now, give the output for the test input.', 'Test Input:', row(test['input']), 'Output:']
    items.append({'dataset': '1d_arc', 'ids': tok('\n'.join(parts))['input_ids'],
                  'split': 'val' if n >= N_ARC else 'train', 'raw': {'test': test}, 'src': f})

for n, it in enumerate(items):
    it['id'] = n
torch.save(items, OUT + '/prompts.pt')
from collections import Counter
lens = [len(it['ids']) for it in items]
print(Counter((it['dataset'], it['split']) for it in items))
print(f'prompt tokens: mean {sum(lens)/len(lens):.0f}, max {max(lens)}, min {min(lens)}')
print('saved', OUT + '/prompts.pt')
