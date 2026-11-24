"""SFT training for Qwen3.5-0.8B-Base on maze / sudoku / countdown / 1d_arc.

Teaches the model the answer format for each task so that pass@k is non-zero
before RL training starts.

Usage:
  python -I sft_train.py --dataset maze     --out checkpoints/maze-sft
  python -I sft_train.py --dataset sudoku   --out checkpoints/sudoku-sft
  python -I sft_train.py --dataset countdown --out checkpoints/countdown-sft
  python -I sft_train.py --dataset 1d_arc   --out checkpoints/arc-sft
  python -I sft_train.py --dataset all      --out checkpoints/all-sft

For --dataset all: mix all four tasks into one training run (recommended
so a single checkpoint handles all domains).
"""
import argparse, csv, json, math, os, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--dataset',   default='all',
                choices=['maze', 'sudoku', 'countdown', '1d_arc', 'all'])
ap.add_argument('--out',       default='/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-all-sft')
ap.add_argument('--epochs',    type=int,   default=1)
ap.add_argument('--lr',        type=float, default=1e-5)
ap.add_argument('--bs',        type=int,   default=8,   help='sequences per step')
ap.add_argument('--grad-accum',type=int,   default=4,   help='accumulation steps')
ap.add_argument('--max-len',   type=int,   default=512, help='max total token length')
ap.add_argument('--warmup',    type=float, default=0.05)
ap.add_argument('--seed',      type=int,   default=333)
ap.add_argument('--eval-every',type=int,   default=200)
ap.add_argument('--probe',     action='store_true')
ap.add_argument('--resume',    action='store_true', help='resume from last.pt')
a = ap.parse_args()

DEV = 'cuda:0'
P   = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'

torch.manual_seed(a.seed)
random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(P)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV)
model.train()

# ── Data builders ─────────────────────────────────────────────────────────────

def load_maze(split='train'):
    path = f'/home/dev/KAP/data/maze_sudoku/maze_{split}.csv'
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['sol']:
                rows.append({'prompt': r['maze'] + '\n\n', 'completion': r['sol']})
    return rows

def load_sudoku(split='train'):
    path = f'/home/dev/KAP/data/maze_sudoku/sudoku_{split}.csv'
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['sol']:
                rows.append({'prompt': r['puzzle'] + '\n\n', 'completion': r['sol']})
    return rows

def load_countdown(split='train'):
    import torch as _t
    items = _t.load('/home/dev/KAP/data/domain_exp/prompts.pt', weights_only=False)
    rows = []
    for it in items:
        if it.get('dataset') != 'countdown': continue
        if it['split'] != split: continue
        raw = it['raw']
        nums, target = raw['nums'], raw['target']
        # Build a valid completion expression (brute-force smallest)
        expr = _make_countdown_expr(nums, target)
        if expr is None: continue
        prompt = f"Target: {target}\nNumbers: {nums}\n"
        rows.append({'prompt': prompt, 'completion': expr})
    return rows

def _make_countdown_expr(nums, target):
    """Find a valid arithmetic expression using all nums that equals target."""
    from fractions import Fraction
    def gen(items):
        if len(items) == 1:
            yield items[0]
            return
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j: continue
                others = [items[k] for k in range(len(items)) if k != i and k != j]
                vi, si = items[i]
                vj, sj = items[j]
                for op, sym in ((vi+vj,f'({si}+{sj})'),
                                (vi-vj,f'({si}-{sj})'),
                                (vi*vj,f'({si}*{sj})'),
                                (vi/vj if vj != 0 else None, f'({si}/{sj})')):
                    if op is None: continue
                    yield from gen(others + [(op, sym)])
    t = Fraction(target)
    for val, expr in gen([(Fraction(n), str(n)) for n in nums]):
        if val == t:
            return expr
    return None

def load_arc(split='train'):
    import torch as _t
    items = _t.load('/home/dev/KAP/data/domain_exp/prompts.pt', weights_only=False)
    rows = []
    for it in items:
        if it.get('dataset') != '1d_arc': continue
        if it['split'] != split: continue
        prompt = tok.decode(it['ids'])
        answer = it['raw']['test']['output'][0]
        completion = str(answer)
        rows.append({'prompt': prompt, 'completion': completion})
    return rows

# ── Load datasets ──────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
loaders = {
    'maze':      load_maze,
    'sudoku':    load_sudoku,
    'countdown': load_countdown,
    '1d_arc':    load_arc,
}

if a.dataset == 'all':
    train_data, val_data = [], []
    for name, fn in loaders.items():
        tr = fn('train'); va = fn('val')
        train_data += tr; val_data += va
        print(f'  {name}: {len(tr)} train, {len(va)} val', flush=True)
else:
    train_data = loaders[a.dataset]('train')
    val_data   = loaders[a.dataset]('val')
    print(f'{a.dataset}: {len(train_data)} train, {len(val_data)} val', flush=True)

random.shuffle(train_data)
print(f'Total train: {len(train_data)}  val: {len(val_data)}', flush=True)

# ── Tokenise + collate ─────────────────────────────────────────────────────────
def tokenise(item):
    """Returns input_ids and loss_mask (1 only on completion tokens)."""
    p_ids = tok(item['prompt'],     add_special_tokens=True)['input_ids']
    c_ids = tok(item['completion'], add_special_tokens=False)['input_ids']
    c_ids = c_ids + [tok.eos_token_id]
    ids   = (p_ids + c_ids)[: a.max_len]
    mask  = ([0] * len(p_ids) + [1] * len(c_ids))[: a.max_len]
    return ids, mask

# Pre-tokenise everything once
print('Tokenising...', flush=True)
train_toks = [tokenise(it) for it in train_data]
val_toks   = [tokenise(it) for it in val_data]
print('Tokenising done.', flush=True)

def collate(batch):
    # batch items are already tokenised tuples (ids, mask)
    ids_list, mask_list = zip(*batch)
    maxlen = max(len(x) for x in ids_list)
    pad    = tok.pad_token_id
    ids_t  = torch.tensor(
        [list(x) + [pad] * (maxlen - len(x)) for x in ids_list], dtype=torch.long)
    mask_t = torch.tensor(
        [list(m) + [0]   * (maxlen - len(m)) for m in mask_list], dtype=torch.long)
    return ids_t, mask_t

# ── Training loop ──────────────────────────────────────────────────────────────
steps_per_epoch = math.ceil(len(train_data) / a.bs)
total_steps     = steps_per_epoch * a.epochs
warmup_steps    = int(a.warmup * total_steps)

opt   = torch.optim.AdamW(model.parameters(), lr=a.lr,
                           betas=(0.9, 0.95), weight_decay=0.01)
def lr_fn(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

log_path = os.path.join(a.out, 'train_log.jsonl')
logf = open(log_path, 'a' if a.resume else 'w')

best_val_loss = float('inf')
t0 = time.time()
step = 0

# Resume from last.pt if requested
if a.resume:
    ckpt_path = os.path.join(a.out, 'last.pt')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        step = ck.get('step', 0)
        best_val_loss = ck.get('best_val_loss', float('inf'))
        print(f'resumed from {ckpt_path}  step={step}  best_val_loss={best_val_loss:.4f}', flush=True)
    else:
        print(f'no last.pt found in {a.out}, starting fresh', flush=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.reset_peak_memory_stats()

@torch.no_grad()
def eval_loss():
    model.eval()
    total, n = 0.0, 0
    rng = random.Random(42)
    sample = rng.sample(val_toks, min(200, len(val_toks)))
    for b0 in range(0, len(sample), a.bs):
        batch = sample[b0: b0 + a.bs]
        ids, mask = collate(batch)
        ids, mask = ids.to(DEV), mask.to(DEV)
        with torch.autocast('cuda', torch.bfloat16):
            logits = model(input_ids=ids).logits[:, :-1]
        targets = ids[:, 1:]
        lmask   = mask[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1), reduction='none')
        total += (ce * lmask.reshape(-1)).sum().item()
        n     += lmask.sum().item()
    model.train()
    return total / max(n, 1)

for epoch in range(a.epochs):
    order = list(range(len(train_toks)))
    random.shuffle(order)
    opt.zero_grad(set_to_none=True)

    for b0 in range(0, len(order), a.bs):
        step += 1
        if step <= (ck.get('step', 0) if a.resume else 0):
            continue
        batch = [train_toks[i] for i in order[b0: b0 + a.bs]]
        ids, mask = collate(batch)
        ids, mask = ids.to(DEV), mask.to(DEV)

        with torch.autocast('cuda', torch.bfloat16):
            logits = model(input_ids=ids).logits[:, :-1]
        targets = ids[:, 1:]
        lmask   = mask[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1), reduction='none')
        loss = (ce * lmask.reshape(-1)).sum() / lmask.sum().clamp(min=1)
        (loss / a.grad_accum).backward()

        if step % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)

        if a.probe and step == 3:
            print(f'probe: peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB  '
                  f'{(time.time()-t0)/3:.2f} s/step  total_steps {total_steps}', flush=True)
            sys.exit(0)

        if step % 50 == 0:
            rec = {'step': step, 'loss': float(loss), 'lr': sched.get_last_lr()[0],
                   'epoch': epoch}
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            print(f'epoch {epoch+1}  step {step:5d}/{total_steps}  '
                  f'loss {float(loss):.4f}  lr {rec["lr"]:.2e}', flush=True)

        if step % a.eval_every == 0 or step == total_steps:
            vl = eval_loss()
            print(f'  val_loss {vl:.4f}', flush=True)
            if vl < best_val_loss:
                best_val_loss = vl
                torch.save({'model': model.state_dict(), 'step': step,
                            'val_loss': vl},
                           os.path.join(a.out, 'best.pt'))
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'step': step,
                        'best_val_loss': best_val_loss},
                       os.path.join(a.out, 'last.pt'))

logf.close()
print(f'done  {round(time.time()-t0)}s  best_val_loss {best_val_loss:.4f}', flush=True)

# Save full HF model so drafter gen-worker can load it as a target
hf_dir = os.path.join(a.out, 'hf_model')
print(f'Saving HF model to {hf_dir} ...', flush=True)
model.save_pretrained(hf_dir)
tok.save_pretrained(hf_dir)
print('HF model saved.', flush=True)
