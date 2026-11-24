'''Generate Target responses for one shard of the prompt list (one process per GPU).
Default is pure temperature sampling (top_p=1.0, top_k=0); pass --top-p/--top-k for the
official recipe (T=0.7, top_p=0.8, top_k=20). Results go to --out (default: gen).'''
import argparse, glob, os, time
import torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--rank', type=int, required=True)
ap.add_argument('--world', type=int, required=True)
ap.add_argument('--work', required=True)
ap.add_argument('--target', required=True)
ap.add_argument('--max-new', type=int, default=512)
ap.add_argument('--temperature', type=float, required=True)
ap.add_argument('--top-p', type=float, default=1.0)
ap.add_argument('--top-k', type=int, default=0)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--ds-filter', default=None,
                help='filter prompts by dataset field (maze/sudoku/countdown/1d_arc)')
ap.add_argument('--out', default='gen')
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.target)
EOS = tok.eos_token_id
IM_END = tok.convert_tokens_to_ids('<|im_end|>')
try:
    model = AutoModelForImageTextToText.from_pretrained(
        a.target, dtype=torch.bfloat16, device_map='cuda:0').eval()
except ValueError:
    model = AutoModelForCausalLM.from_pretrained(
        a.target, dtype=torch.bfloat16, device_map='cuda:0').eval()
print('generation_config:', model.generation_config, flush=True)

items = torch.load(a.work + '/prompts.pt')
if a.ds_filter:
    items = [c for c in items if c.get('dataset', a.ds_filter) == a.ds_filter]
    print(f'ds-filter={a.ds_filter!r}: {len(items)} items', flush=True)
mine = [c for c in items if c['id'] % a.world == a.rank]
if a.limit:
    mine = mine[:a.limit]
out_dir = a.work + '/' + a.out
os.makedirs(out_dir, exist_ok=True)
done = set()
shards = sorted(glob.glob(f'{out_dir}/shard_r{a.rank}_*.pt'))
for f in shards:
    done.update(r['id'] for r in torch.load(f))
mine = [c for c in mine if c['id'] not in done]
mine.sort(key=lambda c: len(c['ids']))
print(f'rank {a.rank}: {len(mine)} prompts to do, {len(done)} already done', flush=True)

buf, shard_k, n_tok, t0 = [], len(shards), 0, time.time()
for b0 in range(0, len(mine), a.bs):
    batch = mine[b0:b0 + a.bs]
    L = max(len(c['ids']) for c in batch)
    ids = torch.full((len(batch), L), EOS, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    for i, c in enumerate(batch):                      # 左 padding
        n = len(c['ids'])
        ids[i, L - n:] = torch.tensor(c['ids'])
        am[i, L - n:] = 1
    with torch.inference_mode():
        gen = model.generate(input_ids=ids.cuda(), attention_mask=am.cuda(), max_new_tokens=a.max_new,
                             do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                             eos_token_id=[IM_END, EOS], pad_token_id=EOS)
    for c, row in zip(batch, gen[:, L:].cpu().tolist()):
        stop = next((j for j, t in enumerate(row) if t in (IM_END, EOS)), None)
        resp = row if stop is None else row[:stop + 1]  # 保留结尾的 im_end / eos
        buf.append({'id': c['id'], 'resp': resp, 'finished': stop is not None})
        n_tok += len(resp)
    last = b0 + a.bs >= len(mine)
    if len(buf) >= a.bs * 8 or last:
        tmp = f'{out_dir}/shard_r{a.rank}_{shard_k:04d}.pt.tmp'
        torch.save(buf, tmp)
        os.replace(tmp, tmp[:-4])                      # 原子落盘，崩溃不会留下半个 shard
        shard_k, buf = shard_k + 1, []
    print(f'rank {a.rank}: {min(b0 + a.bs, len(mine))}/{len(mine)}  {n_tok / (time.time() - t0):.0f} tok/s', flush=True)
