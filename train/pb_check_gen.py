'''Exit 0 only if every prompt in prompts.pt has a generated response on disk.'''
import glob, sys
import torch

W = '/home/dev/KAP/data/pb_dspark'
items = torch.load(W + '/prompts.pt')
ids = set()
for f in glob.glob(W + '/gen/shard_r*_*.pt'):
    ids.update(r['id'] for r in torch.load(f))
need = {c['id'] for c in items}
miss = need - ids
print('responses', len(ids & need), 'of', len(need), 'missing', len(miss))
sys.exit(0 if not miss else 1)
