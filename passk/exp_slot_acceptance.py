"""Per-draft-slot acceptance rate by sampling setting.

For each (domain, mode): run generate_branches(B=8) over N val prompts and record,
for every verify round, the number of accepted drafts k (first rejection index).
For slot j = 1..n:
    reached_j = #rounds with k >= j-1   (all j-1 earlier slots accepted)
    accept_j  = #rounds with k >= j
    alpha_j   = accept_j / reached_j    (acceptance rate AT slot j)
This is the end-to-end (conditional-on-reaching) acceptance curve.

Modes (inverse CDF only): iid, stratified, stratified_fixed.
"""
from __future__ import annotations
import argparse, json, random, sys, time
import torch
sys.path.insert(0, '/home/dev/KAP/src4dspark')
import dspark_fast, dspark_paper
from dspark_paper import DSparkDrafter, ND
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--domains', nargs='+', default=['sudoku', '1d_arc'])
ap.add_argument('--n-prompts', type=int, default=40)
ap.add_argument('--branches', type=int, default=8)
ap.add_argument('--device', default='cuda:1')
ap.add_argument('--out', default='/home/dev/KAP/passk/exp_slot_acceptance.json')
args = ap.parse_args()

PAR = '/home/dev/KAP/parameter'
CFG = {
    'maze':      dict(target=PAR+'/target/Qwen3.5-0.8B-Base-maze-sft/hf_model',
                      drafter=PAR+'/draft/dspark_maze_sft/best.pt',
                      data='/home/dev/KAP/data/maze_sudoku/prompts.pt', max_new=64),
    'sudoku':    dict(target=PAR+'/target/Qwen3.5-0.8B-Base-sudoku-sft/hf_model',
                      drafter=PAR+'/draft/dspark_sudoku_sft/best.pt',
                      data='/home/dev/KAP/data/maze_sudoku/prompts.pt', max_new=300),
    'countdown': dict(target=PAR+'/target/Qwen3.5-0.8B-Base-countdown-sft/hf_model',
                      drafter=PAR+'/draft/dspark_countdown_sft/best.pt',
                      data='/home/dev/KAP/data/domain_exp/prompts.pt', max_new=64),
    '1d_arc':    dict(target=PAR+'/target/Qwen3.5-0.8B-Base-original',
                      drafter=PAR+'/draft/dspark_domain/best.pt',
                      data='/home/dev/KAP/data/domain_exp/prompts.pt', max_new=320),
}
MODES = ('iid', 'stratified', 'stratified_fixed')



class FixedOrderEngine(dspark_fast.DSparkDecodeFast):
    """Pre-fix behaviour: fixed stratum order (diagnostic only)."""

    @torch.inference_mode()
    def _propose_multi(self, feats1, anchor, anchor_pos, n, B, mode='stratified_fixed'):
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        o = torch.tensor([[anchor_pos - 1]], device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anchor.view(1, 1)), o)[0, 0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()
        v = torch.rand(B, device=self.dev, generator=self.gen)
        u = (torch.arange(B, device=self.dev, dtype=torch.float32) + v) / B
        prev = anchor.view(1).expand(B)
        Lg = torch.empty(B, n, V, device=self.dev)
        ids = torch.empty(B, n, dtype=torch.long, device=self.dev)
        lps = torch.zeros(B, device=self.dev)
        for j in range(n):
            lgj = (base[j].unsqueeze(0) + self.d.markov(prev)) / self.T
            Lg[:, j] = lgj
            q = torch.softmax(lgj, -1)
            cdf = q.cumsum(-1)
            xj = torch.searchsorted(cdf, u.unsqueeze(1), right=True).squeeze(1).clamp(max=V - 1)
            qx = q.gather(1, xj.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            lo = cdf.gather(1, xj.unsqueeze(1)).squeeze(1) - qx
            lps += torch.log(qx)
            u = ((u - lo) / qx).clamp(0.0, 1.0 - 1e-7)
            ids[:, j] = xj
            prev = xj
        return Lg, ids, lps, u


def load_prompts(cfg, ds, n, seed=42):
    items = torch.load(cfg['data'], map_location='cpu', weights_only=False)
    val = [it for it in items if it.get('dataset') == ds and it['split'] == 'val']
    return random.Random(seed).sample(val, min(n, len(val)))


def run_mode(model, drafter, tok, prompts, mode, max_new, B, seed=42):
    """Return per-slot reach/accept counts accumulated over all rounds."""
    reach = [0] * (ND + 1)      # index j-1 -> rounds that reached slot j
    accept = [0] * (ND + 1)     # index j-1 -> rounds that accepted slot j
    for it in prompts:
        if mode == 'stratified_fixed':
            eng = FixedOrderEngine(model, drafter, tok, temperature=1.0, seed=seed)
        else:
            eng = dspark_fast.DSparkDecodeFast(model, drafter, tok, temperature=1.0, seed=seed)
        with torch.inference_mode():
            chains = eng.generate_branches(it['ids'], max_new, n=ND, branches=B,
                                           mode=mode, return_all=True)
        for _, st in chains:
            for k in st['accepted']:
                for j in range(1, ND + 1):
                    if k >= j - 1:
                        reach[j - 1] += 1
                        if k >= j:
                            accept[j - 1] += 1
        del eng
        torch.cuda.empty_cache()
    alpha = [accept[j] / reach[j] if reach[j] else None for j in range(ND)]
    return dict(reach=reach[:ND], accept=accept[:ND], alpha=alpha)


def main():
    results = {}
    for ds in args.domains:
        cfg = CFG[ds]
        print(f'\n{"="*60}\n  {ds.upper()}  (B={args.branches}, {args.n_prompts} prompts)\n{"="*60}', flush=True)
        model = AutoModelForCausalLM.from_pretrained(cfg['target'], dtype=torch.bfloat16,
                                                     device_map=args.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(cfg['target'])
        d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128,
                          device=args.device, dtype=torch.float32)
        ck = torch.load(cfg['drafter'], map_location=args.device, weights_only=False)
        d.load_state_dict(ck['model']); d.eval()
        prompts = load_prompts(cfg, ds, args.n_prompts)
        results[ds] = {}
        for mode in MODES:
            t0 = time.perf_counter()
            r = run_mode(model, d, tok, prompts, mode, cfg['max_new'], args.branches)
            results[ds][mode] = r
            a = ' '.join(f'{x:.2f}' if x is not None else '--' for x in r['alpha'])
            print(f'  [{mode:16s}] α(slot1..7) {a}   ({time.perf_counter()-t0:.0f}s)', flush=True)
        del d, model
        torch.cuda.empty_cache()
    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
