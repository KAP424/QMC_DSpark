'''Gap analysis, experiment 2: does more supervision per target forward help?

DeepSpec samples up to 512 anchors per sequence (nearly every response position); pb_train.py
samples 8. Train two drafters from scratch with everything equal (same sequences per step, steps,
LR schedule, seed, data order) except anchors per sequence, then evaluate on the same held-out blocks.
Target frozen; nothing is written under /home/dev/KAP/parameter.
'''
import argparse, glob, json, random, sys, time
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_paper import DSparkDrafter, ND  # noqa: E402
from dspark_loss import make_blocks, dspark_loss, evaluate, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--anchors', type=int, required=True)
ap.add_argument('--seqs', type=int, default=16)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--eval-every', type=int, default=500)
ap.add_argument('--probe', action='store_true', help='run 3 steps and report peak memory only')
ap.add_argument('--out', required=True)
a = ap.parse_args()
DEV = 'cuda:0'
P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
W = '/home/dev/KAP/data/pb_dspark'
torch.backends.cuda.matmul.allow_tf32 = True

tok = AutoTokenizer.from_pretrained(P)
EOS = tok.eos_token_id
items = {c['id']: c for c in torch.load(W + '/prompts.pt')}
resp = {}
for f in sorted(glob.glob(W + '/gen/shard_r*_*.pt')):
    for r in torch.load(f):
        resp[r['id']] = r
data = {'train': [], 'val': []}
for i in sorted(resp):
    if len(resp[i]['resp']) >= 16:
        data[items[i]['split']].append({'id': i, 'prompt_ids': items[i]['ids'], 'resp_ids': resp[i]['resp']})
train, val = data['train'], data['val'][:200]

model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
torch.manual_seed(0)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
params = [p for p in d.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps, pct_start=0.03)

rng = random.Random(0)                       # identical sequence order in every run
order = list(range(len(train)))
rng.shuffle(order)
gen = torch.Generator().manual_seed(0)
log = {'anchors': a.anchors, 'seqs': a.seqs, 'steps': a.steps, 'blocks_per_step': a.anchors * a.seqs, 'evals': []}

def do_eval(step):
    e = evaluate(d, model, EOS, val, 1.0)
    v = summarize(e)
    log['evals'].append({'step': step, 'tau': v['tau'], 'tau_argmax': v['tau_argmax'], 'ce': v['ce'], 'tv': v['tv'],
                         'alpha': e['alpha'].tolist()})
    print(f"step {step:5d}  val tau {v['tau']:.3f}  tau_argmax {v['tau_argmax']:.3f}  CE {v['ce']:.3f}  TV {v['tv']:.3f}  "
          f"alpha {' '.join(f'{x:.2f}' for x in e['alpha'].tolist())}", flush=True)

d.train()
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
for step in range(1, a.steps + 1):
    b0 = ((step - 1) * a.seqs) % (len(order) - a.seqs)
    batch = [train[i] for i in order[b0:b0 + a.seqs]]
    feats, toks, o, p_t = make_blocks(model, EOS, batch, a.anchors, gen, 1.0, DEV)
    loss, st = dspark_loss(d, feats, toks, o, p_t, 1.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    if a.probe and step == 3:
        print(f'probe: anchors {a.anchors} x seqs {a.seqs}: peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, '
              f'{(time.time() - t0) / 3:.2f} s/step (incl. warm-up)', flush=True)
        sys.exit(0)
    if step % a.eval_every == 0 or step == a.steps:
        d.eval(); do_eval(step); d.train()
        log['sec_per_step'] = (time.time() - t0) / step
        log['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
        json.dump(log, open(a.out, 'w'), indent=2)
print('done', round(time.time() - t0), 's', flush=True)
