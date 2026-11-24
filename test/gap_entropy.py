'''Gap analysis, experiment 1: how predictable is the target that the drafter has to match?

The drafter is scored against the target's T=1 distribution (paper and DeepSpec both use T=1 for
the loss and for evaluation). A peaked target distribution is easy to match; a flat one is not.
This compares, on the same 200 held-out prompts:

  A  Base model,     responses sampled at T=1.0 pure          <- what I trained and evaluated on
  B  Base model,     responses sampled at T=0.7/top_p 0.8/top_k 20 (DeepSpec's data params)
  C  Instruct model, responses sampled at T=1.0 pure          <- the paper's evaluation protocol
  D  Instruct model, responses sampled at T=0.7/top_p 0.8/top_k 20 (DeepSpec's training data)

For each, the generating model's own T=1 next-token entropy and max probability on the response.
Then the current drafter's slot-1 acceptance is bucketed by target entropy on A, and re-weighted
by each condition's entropy histogram. That re-weighting is a rough transfer estimate only.
'''
import glob, json, os, sys, time
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

BASE = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
INSTRUCT = '/home/dev/models/Qwen3.5-0.8B'
W = '/home/dev/KAP/data/pb_dspark'
DEV = 'cuda:0'
N, MAX_NEW, BS = 200, 512, 50
EDGES = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 99.0]
torch.manual_seed(0)

items = torch.load(W + '/prompts.pt')
resp = {}
for f in sorted(glob.glob(W + '/gen/shard_r*_*.pt')):
    for r in torch.load(f):
        resp[r['id']] = r
val = [c for c in items if c['split'] == 'val' and len(resp[c['id']]['resp']) >= 16][:N]
tok = AutoTokenizer.from_pretrained(BASE)
EOS, IM_END = tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')


def load(path):
    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()


@torch.inference_mode()
def generate(m, prompts, **kw):
    out = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for b0 in range(0, len(order), BS):
        idx = order[b0:b0 + BS]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), EOS, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i]); am[r, L - len(prompts[i]):] = 1
        g = m.generate(input_ids=ids.to(DEV), attention_mask=am.to(DEV), max_new_tokens=MAX_NEW, do_sample=True,
                       eos_token_id=[IM_END, EOS], pad_token_id=EOS, **kw)
        for r, i in enumerate(idx):
            row = g[r, L:].tolist()
            stop = next((j for j, t in enumerate(row) if t in (IM_END, EOS)), None)
            res[i] = row if stop is None else row[:stop + 1]
    return res


@torch.inference_mode()
def entropy_stats(m, prompts, responses):
    ent, pmax = [], []
    for p, r in zip(prompts, responses):
        if len(r) < 2:
            continue
        ids = torch.tensor([p + r], device=DEV)
        h = m(input_ids=ids, use_cache=False, output_hidden_states=True, logits_to_keep=1).hidden_states[-1][0]
        hh = h[len(p) - 1:len(p) + len(r) - 1]                  # positions predicting the response tokens
        for c0 in range(0, hh.shape[0], 256):
            lp = torch.log_softmax(m.lm_head(hh[c0:c0 + 256]).float(), -1)
            ent.append(-(lp.exp() * lp).sum(-1).cpu()); pmax.append(lp.max(-1).values.exp().cpu())
    ent, pmax = torch.cat(ent), torch.cat(pmax)
    shares = [float(((ent >= lo) & (ent < hi)).float().mean()) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    return dict(tokens=int(ent.numel()), mean_entropy=float(ent.mean()), median_entropy=float(ent.median()),
                mean_pmax=float(pmax.mean()), frac_pmax_gt_0_9=float((pmax > 0.9).float().mean()),
                frac_entropy_gt_2=float((ent > 2).float().mean()), shares=shares)


def resp_stats(responses):
    n = len(responses)
    return dict(mean_len=sum(map(len, responses)) / n,
                finished=sum(1 for r in responses if r and r[-1] in (IM_END, EOS)) / n)


t0 = time.time()
prompts_base = [c['ids'] for c in val]
inst_tok = AutoTokenizer.from_pretrained(INSTRUCT)
prompts_inst = [inst_tok(inst_tok.apply_chat_template([{'role': 'user', 'content': c['prompt']}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False),
                         add_special_tokens=False)['input_ids'] for c in val]
same_template = sum(a == b for a, b in zip(prompts_base, prompts_inst))
print(f'{len(val)} prompts; instruct template identical to base rendering for {same_template}/{len(val)}', flush=True)

OFFICIAL = dict(temperature=0.7, top_p=0.8, top_k=20)
PURE = dict(temperature=1.0, top_p=1.0, top_k=0)
results = {}
base = load(BASE)
resp_A = [resp[c['id']]['resp'][:MAX_NEW] for c in val]
results['A base T1.0'] = dict(**entropy_stats(base, prompts_base, resp_A), **resp_stats(resp_A))
resp_B = generate(base, prompts_base, **OFFICIAL)
results['B base T0.7/p0.8/k20'] = dict(**entropy_stats(base, prompts_base, resp_B), **resp_stats(resp_B))
print('base done', round(time.time() - t0), 's', flush=True)

# drafter slot acceptance bucketed by target entropy, on condition A (the drafter's own distribution)
from dspark_paper import DSparkDrafter, load as dload, ND  # noqa: E402
from dspark_loss import make_blocks  # noqa: E402
d = DSparkDrafter(base, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
dload(d, '/home/dev/KAP/parameter/draft/dspark_pb/dflash_pb_best.pt'); d.eval()
seqs = [{'prompt_ids': c['ids'], 'resp_ids': resp[c['id']]['resp']} for c in val]
al_all, ent_all = [], []
g = torch.Generator().manual_seed(0)
with torch.inference_mode():
    for b0 in range(0, len(seqs), 8):
        feats, toks, o, p_t = make_blocks(base, EOS, seqs[b0:b0 + 8], 16, g, 1.0, DEV)
        z = d.backbone(d.encode_ctx(feats), d.embed(toks[..., 0]).float(), o)
        q = torch.softmax(d.logits(z, toks[..., :ND]), -1)
        al_all.append(torch.minimum(p_t, q).sum(-1).reshape(-1, ND).cpu())
        ent_all.append((-(p_t * torch.log(p_t.clamp_min(1e-30))).sum(-1)).reshape(-1, ND).cpu())
al_all, ent_all = torch.cat(al_all), torch.cat(ent_all)
bucket_alpha, bucket_share = [], []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    m_ = (ent_all[:, 0] >= lo) & (ent_all[:, 0] < hi)
    bucket_alpha.append(float(al_all[m_, 0].mean()) if m_.any() else float('nan'))
    bucket_share.append(float(m_.float().mean()))
results['drafter_slot1_by_entropy'] = dict(edges=EDGES, alpha=bucket_alpha, share_on_A_blocks=bucket_share,
                                           slot1_alpha=float(al_all[:, 0].mean()), blocks=int(al_all.shape[0]))
del d, base
torch.cuda.empty_cache()

inst = load(INSTRUCT)
resp_C = generate(inst, prompts_inst, **PURE)
results['C instruct T1.0'] = dict(**entropy_stats(inst, prompts_inst, resp_C), **resp_stats(resp_C))
resp_D = generate(inst, prompts_inst, **OFFICIAL)
results['D instruct T0.7/p0.8/k20'] = dict(**entropy_stats(inst, prompts_inst, resp_D), **resp_stats(resp_D))
print('instruct done', round(time.time() - t0), 's', flush=True)

ba = results['drafter_slot1_by_entropy']['alpha']
for k in ('A base T1.0', 'B base T0.7/p0.8/k20', 'C instruct T1.0', 'D instruct T0.7/p0.8/k20'):
    sh = results[k]['shares']
    results[k]['reweighted_slot1_alpha'] = sum(s * a for s, a in zip(sh, ba) if a == a)

samples = {'A': tok.decode(resp_A[3][:80]), 'B': tok.decode(resp_B[3][:80]),
           'C': tok.decode(resp_C[3][:80]), 'D': tok.decode(resp_D[3][:80]), 'prompt': val[3]['prompt'][:200]}
json.dump(dict(results=results, samples=samples), open(W + '/gap_entropy.json', 'w'), indent=2, ensure_ascii=False)

print()
print(f"{'condition':28s} {'tokens':>7s} {'H mean':>7s} {'H med':>6s} {'pmax':>6s} {'p>0.9':>6s} {'H>2':>6s} {'len':>6s} {'fin':>5s} {'slot1*':>7s}")
for k in ('A base T1.0', 'B base T0.7/p0.8/k20', 'C instruct T1.0', 'D instruct T0.7/p0.8/k20'):
    r = results[k]
    print(f"{k:28s} {r['tokens']:7d} {r['mean_entropy']:7.3f} {r['median_entropy']:6.3f} {r['mean_pmax']:6.3f} "
          f"{r['frac_pmax_gt_0_9']:6.3f} {r['frac_entropy_gt_2']:6.3f} {r['mean_len']:6.1f} {r['finished']:5.2f} {r['reweighted_slot1_alpha']:7.3f}")
print()
print('drafter slot-1 alpha by target entropy bucket (condition A blocks, n=%d, overall %.3f):' % (
    results['drafter_slot1_by_entropy']['blocks'], results['drafter_slot1_by_entropy']['slot1_alpha']))
for lo, hi, a, s in zip(EDGES[:-1], EDGES[1:], ba, results['drafter_slot1_by_entropy']['share_on_A_blocks']):
    print(f'  H in [{lo:4.1f},{hi:4.1f})  share {s:5.3f}  alpha {a:5.3f}')
print('* slot1 = drafter slot-1 alpha re-weighted by that condition\'s entropy histogram (rough transfer estimate)')
print('total', round(time.time() - t0), 's')
