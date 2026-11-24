'''Background trainer for DSparkDrafter. Metrics go to <work>/train_log.jsonl (one JSON per line) so the notebook can plot them live.'''
import argparse, glob, hashlib, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src4dspark'))
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, save
from dspark_loss import make_blocks, dspark_loss, summarize, evaluate, ND

ap = argparse.ArgumentParser()
ap.add_argument('--work', default='/home/dev/KAP/data/pb_dspark')
ap.add_argument('--target', default='/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original')
ap.add_argument('--ckpt', default='/home/dev/KAP/parameter/draft/dspark_pb/dflash_pb.pt')
ap.add_argument('--state', default='/home/dev/KAP/parameter/draft/dspark_pb/train_state.pt')
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--temp', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seqs', type=int, default=16)
ap.add_argument('--anchors', type=int, default=8)
ap.add_argument('--log-every', type=int, default=50)
ap.add_argument('--eval-every', type=int, default=500)
ap.add_argument('--min-resp', type=int, default=16)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
DEV = 'cuda:0'
LOG, STATE = a.work + '/train_log.jsonl', a.state
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + chr(10))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 26), b''):
            h.update(blk)
    return h.hexdigest()

tok = AutoTokenizer.from_pretrained(a.target)
EOS = tok.eos_token_id
items = {c['id']: c for c in torch.load(a.work + '/prompts.pt')}
resp = {}
for f in sorted(glob.glob(a.work + '/gen/shard_r*_*.pt')):
    for r in torch.load(f):
        resp[r['id']] = r
data = {'train': [], 'val': []}
for i in sorted(resp):
    r, c = resp[i], items[i]
    if len(r['resp']) >= a.min_resp:
        data[c['split']].append({'id': i, 'source': c['source'], 'prompt_ids': c['ids'], 'resp_ids': r['resp']})
train, val = data['train'], data['val']
print('train', len(train), 'val', len(val), flush=True)

weight_file = a.target + '/model.safetensors-00001-of-00001.safetensors'
sha0 = sha256(weight_file)
model = AutoModelForImageTextToText.from_pretrained(a.target, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
assert not ({id(p) for p in model.parameters()} & {id(p) for p in d.parameters()}), 'target leaked into drafter'
params = [p for p in d.parameters() if p.requires_grad]
spe = len(train) // a.seqs
total = a.epochs * spe
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.03)
step0 = 0
if os.path.exists(STATE):
    ck = torch.load(STATE, map_location=DEV, weights_only=False)
    d.load_state_dict(ck['d']); opt.load_state_dict(ck['opt'])
    step0 = ck['step']
    if ck['sched']['total_steps'] == total:
        sched.load_state_dict(ck['sched'])
    else:
        # total steps changed (e.g. 5 -> 10 epochs): the saved schedule cannot be stepped
        # past its own end, so rebuild it for the new length and fast-forward to step0.
        for _ in range(step0):
            sched.step()
        print(f'rebuilt schedule: {ck["sched"]["total_steps"]} -> {total} steps, fast-forwarded to {step0}', flush=True)
    print('resumed from step', step0, flush=True)
if step0 == 0:
    log({'t': 'config', **vars(a), 'n_train': len(train), 'n_val': len(val), 'steps_per_epoch': spe, 'total_steps': total, 'params': d.n_params, 'sha_before': sha0, 'time': time.time()})
else:
    log({'t': 'config', **vars(a), 'n_train': len(train), 'n_val': len(val), 'steps_per_epoch': spe, 'total_steps': total, 'params': d.n_params, 'resumed_from': step0, 'time': time.time()})

def val_eval(step):
    v = evaluate(d, model, EOS, val[:200], a.temp)
    r = summarize(v)
    log({'t': 'val', 'step': step, 'epoch': step / spe, **r, 'alpha': v['alpha'].tolist(), 'conf': v['conf'].tolist(), 'hit': v['hit'].tolist()})
    print('val', step, {k: round(x, 3) for k, x in r.items()}, flush=True)
    return r['tau']

def batch_iter(seqs, bs, rng):
    while True:
        perm = list(range(len(seqs)))
        rng.shuffle(perm)
        for m0 in range(0, len(perm), bs * 32):
            chunk = sorted(perm[m0:m0 + bs * 32], key=lambda i: len(seqs[i]['prompt_ids']) + len(seqs[i]['resp_ids']))
            bl = [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
            rng.shuffle(bl)
            for b in bl:
                if len(b) == bs:
                    yield [seqs[i] for i in b]

if step0 == 0:
    best = val_eval(0)
else:
    # resuming: keep the best tau already reached, so a worse checkpoint cannot overwrite the best one
    prev = [json.loads(l) for l in open(LOG) if l.strip().endswith('}')] if os.path.exists(LOG) else []
    best = max([r['tau'] for r in prev if r['t'] == 'val'], default=-1.0)
    print('previous best val tau', best, flush=True)
d.train()
rng, gen = random.Random(a.seed + step0), torch.Generator().manual_seed(a.seed + step0)
it = batch_iter(train, a.seqs, rng)
acc, n, wl, t0 = None, 0, 0.0, time.time()
for step in range(step0 + 1, total + 1):
    feats, toks, o, p_t = make_blocks(model, EOS, next(it), a.anchors, gen, a.temp, DEV)
    loss, st = dspark_loss(d, feats, toks, o, p_t, a.temp)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    wl += loss.item(); n += 1
    acc = st if acc is None else {k: acc[k] + st[k] for k in st}
    if step % a.log_every == 0:
        r = summarize({k: v / n for k, v in acc.items()})
        log({'t': 'train', 'step': step, 'epoch': step / spe, 'loss': wl / n, **r, 'alpha': (acc['alpha'] / n).tolist(), 'lr': sched.get_last_lr()[0], 'ms': (time.time() - t0) / (step - step0) * 1e3})
        acc, n, wl = None, 0, 0.0
    if step % a.eval_every == 0 or step == total:
        tau = val_eval(step)
        save(d, a.ckpt)
        torch.save({'d': d.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, STATE)
        if tau > best:
            best = tau
            save(d, a.ckpt.replace('.pt', '_best.pt'))

sha1 = sha256(weight_file)
log({'t': 'done', 'sha_before': sha0, 'sha_after': sha1, 'target_unchanged': sha0 == sha1, 'best_val_tau': best, 'time': time.time()})
print('done; target unchanged:', sha0 == sha1, ' best val tau', best, flush=True)
