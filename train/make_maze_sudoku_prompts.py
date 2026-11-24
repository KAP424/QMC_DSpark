"""Convert maze + sudoku CSV files to prompts.pt (same schema as pb_dspark/prompts.pt).
Columns used: maze/puzzle -> prompt, sol -> ground-truth answer.
Run after gen_maze.py / gen_sudoku.py finish."""
import csv, os, sys, torch
from transformers import AutoTokenizer

P   = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
OUT = '/home/dev/KAP/data/maze_sudoku'
tok = AutoTokenizer.from_pretrained(P)

SPLITS = [
    ('maze',   OUT + '/maze_train.csv',   'train'),
    ('maze',   OUT + '/maze_val.csv',     'val'),
    ('sudoku', OUT + '/sudoku_train.csv', 'train'),
    ('sudoku', OUT + '/sudoku_val.csv',   'val'),
]

items = []
for dataset, path, split in SPLITS:
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        sys.exit(1)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    pkey = 'maze' if dataset == 'maze' else 'puzzle'
    for row in rows:
        prompt = row[pkey] + '\n\n'
        items.append({
            'id': len(items),
            'dataset': dataset,
            'split': split,
            'ids': tok(prompt)['input_ids'],
            'raw': {'sol': row['sol']},
        })
    print(f'{dataset}/{split}: {len(rows)} rows loaded')

torch.save(items, OUT + '/prompts.pt')
from collections import Counter
lens = [len(it['ids']) for it in items]
print(Counter((it['dataset'], it['split']) for it in items))
print(f'prompt tokens: mean {sum(lens)/len(lens):.0f}, max {max(lens)}, min {min(lens)}')
print('saved', OUT + '/prompts.pt')
