"""Run Maze (100 prompts x 64 rollouts) and Speedup benchmark across the 3 S5 variants on GPU:
  - S5_update    : Draft uses inverse-CDF and updates draft u via q-residual; verify uses post-anchor u_b.
  - S5_no_update : Draft uses inverse-CDF with fixed post-anchor u_b (no draft residual update); verify uses post-anchor u_b.
  - S5_gumbel    : Draft uses Gumbel-Max random noise (no u in draft); verify uses post-anchor u_b.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re, sys, time
from pathlib import Path
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
import dspark_fast
from dspark_paper import DSparkDrafter, ND
from transformers import AutoModelForCausalLM, AutoTokenizer

PAR = '/home/dev/KAP/parameter'
OUT = Path('/home/dev/KAP/passk/sample')
B = 64
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
T = 1.0
SEED = 42


def lattice(Bn, gen, dev):
    d = torch.rand((), device=dev, generator=gen)
    return (d + torch.arange(Bn, device=dev, dtype=torch.float32) / Bn) % 1.0


def lattice_est(hits, k):
    N = len(hits)
    stride = N // k
    vals = [1.0 if any(hits[j] for j in range(r, N, stride)) else 0.0 for r in range(stride)]
    return sum(vals) / len(vals)


class S5VariantEngine(dspark_fast.DSparkDecodeFast):
    def __init__(self, *a, draft_variant='update', **kw):
        super().__init__(*a, **kw)
        self.draft_variant = draft_variant

    @torch.inference_mode()
    def _propose_multi(self, feats1, anchor, anchor_pos, n, B, u=None, mode='stratified'):
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        if not torch.is_tensor(anchor):
            anchor = torch.tensor(anchor, device=self.dev)
        if anchor.numel() == 1:
            anc = anchor.reshape(1, 1).expand(1, B)
        else:
            anc = anchor.view(1, B)
        o = torch.full((1, B), anchor_pos - 1, dtype=torch.long, device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anc), o)[0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()

        prev = anc.view(B)
        Lg = torch.empty(B, n, V, device=self.dev)
        ids = torch.empty(B, n, dtype=torch.long, device=self.dev)
        lps = torch.zeros(B, device=self.dev)

        if self.draft_variant == 'gumbel':
            for j in range(n):
                lgj = (base[:, j] + self.d.markov(prev)) / self.T
                Lg[:, j] = lgj
                unif = torch.rand(B, V, device=self.dev, generator=self.gen).clamp(1e-7, 1.0 - 1e-7)
                gumbel = -torch.log(-torch.log(unif))
                xj = (lgj + gumbel).argmax(dim=-1)
                ids[:, j] = xj
                prev = xj
            return Lg, ids, lps, u

        u_cur = u.to(self.dev).float().view(B).clone()
        for j in range(n):
            lgj = (base[:, j] + self.d.markov(prev)) / self.T
            Lg[:, j] = lgj
            q = torch.softmax(lgj, -1)
            cdf = q.cumsum(-1)
            xj = torch.searchsorted(cdf, u_cur.unsqueeze(1), right=True).squeeze(1).clamp(max=V - 1)
            qx = q.gather(1, xj.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            lps += torch.log(qx)
            if self.draft_variant == 'update':
                lo = cdf.gather(1, xj.unsqueeze(1)).squeeze(1) - qx
                u_cur = ((u_cur - lo) / qx).clamp(0.0, 1.0 - 1e-7)
            ids[:, j] = xj
            prev = xj
        return Lg, ids, lps, u

    @torch.inference_mode()
    def _propose_single(self, feats1, anchor, anchor_pos, n, u, mode='stratified'):
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        o = torch.tensor([[anchor_pos - 1]], device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anchor.view(1, 1)), o)[0, 0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()
        prev = anchor.clone() if isinstance(anchor, torch.Tensor) else torch.tensor(anchor, device=self.dev)
        Lg = torch.empty(n, V, device=self.dev)
        ids = torch.empty(n, dtype=torch.long, device=self.dev)
        lps = torch.zeros((), device=self.dev)

        if self.draft_variant == 'gumbel':
            for j in range(n):
                lgj = (base[j] + self.d.markov(prev.view(1))[0]) / self.T
                Lg[j] = lgj
                unif = torch.rand(V, device=self.dev, generator=self.gen).clamp(1e-7, 1.0 - 1e-7)
                gumbel = -torch.log(-torch.log(unif))
                xj = int((lgj + gumbel).argmax())
                ids[j] = xj
                prev = torch.tensor(xj, device=self.dev)
            return Lg, ids, lps, u

        uu = float(u)
        for j in range(n):
            lgj = (base[j] + self.d.markov(prev.view(1))[0]) / self.T
            Lg[j] = lgj
            q = torch.softmax(lgj, -1)
            cdf = q.cumsum(-1)
            xj = int(torch.searchsorted(cdf, torch.tensor(uu, device=self.dev), right=True).clamp(max=V - 1))
            qx = float(q[xj].clamp_min(1e-12))
            lps = lps + torch.log(q[xj])
            if self.draft_variant == 'update':
                lo = float(cdf[xj]) - qx
                uu = min(max((uu - lo) / qx, 0.0), 1.0 - 1e-7)
            ids[j] = xj
            prev = torch.tensor(xj, device=self.dev)
        return Lg, ids, lps, u


def check_maze(resp: str, grid_str: str) -> bool:
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.S | re.I)
    if m:
        resp = m.group(1)
    path = resp.strip()
    if not path or any(c not in 'URDL' for c in path):
        return False
    g = [list(r) for r in grid_str.strip().splitlines()]
    H, W = len(g), len(g[0])
    si = sc = ei = ec = None
    for i in range(H):
        for j in range(W):
            if g[i][j] == 'S':
                si, sc = i, j
            elif g[i][j] == 'E':
                ei, ec = i, j
    r, c = si, sc
    dd = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    for ch in path:
        dr, dc = dd[ch]
        r, c = r + dr, c + dc
        if not (0 <= r < H and 0 <= c < W) or g[r][c] == '*':
            return False
    return (r, c) == (ei, ec)


@torch.inference_mode()
def baseline_one(model, tok, prompt_ids, max_new, dev):
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    ids = torch.tensor([prompt_ids], device=dev)
    gen = model.generate(input_ids=ids, max_new_tokens=max_new,
                         do_sample=True, temperature=1.0,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    return int(gen.shape[1] - ids.shape[1]), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['update', 'no_update', 'gumbel'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-prompts', type=int, default=100)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = args.device
    target_path = PAR + '/target/Qwen3.5-0.8B-Base-maze-sft/hf_model'
    drafter_path = PAR + '/draft/dspark_maze_sft/best.pt'
    max_new = 64

    tok = AutoTokenizer.from_pretrained(target_path)
    model = AutoModelForCausalLM.from_pretrained(target_path, dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    drafter = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=dev, dtype=torch.float32)
    drafter.load_state_dict(torch.load(drafter_path, map_location=dev, weights_only=False)['model'])
    drafter.eval()

    items = torch.load('/home/dev/KAP/data/maze_sudoku/prompts.pt', map_location='cpu', weights_only=False)
    val = [it for it in items if it.get('dataset') == 'maze' and it['split'] == 'val']
    rows = list(csv.DictReader(open('/home/dev/KAP/data/maze_sudoku/maze_val.csv')))
    val.sort(key=lambda x: x['id'])
    base_id = val[0]['id']
    val = [dict(it, maze_grid=rows[it['id'] - base_id]['maze'])
           for it in val if 0 <= it['id'] - base_id < len(rows)]
    prompts = random.Random(SEED + 1).sample(val, min(args.n_prompts, len(val)))

    # 1) Measure B=1 tok/s speedup & tau on 50 prompts
    sp_prompts = prompts[:50]
    baseline_one(model, tok, sp_prompts[0]['ids'], max_new, dev)
    eng_sp = S5VariantEngine(model, drafter, tok, temperature=T, seed=SEED, draft_variant=args.variant)
    eng_sp.generate(sp_prompts[0]['ids'], max_new, n=ND, branches=1, mode='lattice', u0=0.37)

    base_toks, base_dt = 0, 0.0
    ds_toks, ds_dt = 0, 0.0
    b1_accs = []
    for idx, it in enumerate(sp_prompts):
        nb, db = baseline_one(model, tok, it['ids'], max_new, dev)
        base_toks += nb
        base_dt += db

        u_single = ((0.173 + idx * 0.61803398875) % 1.0)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        seq_out, st = eng_sp.generate(it['ids'], max_new, n=ND, branches=1, mode='lattice', u0=u_single)
        torch.cuda.synchronize(dev)
        ds_dt += (time.perf_counter() - t0)
        ds_toks += len(seq_out)
        b1_accs.extend(st['accepted'])

    base_tps = base_toks / base_dt
    ds_tps = ds_toks / ds_dt
    speedup_b1 = ds_tps / base_tps
    tau_b1 = (sum(b1_accs) / len(b1_accs) + 1.0) if b1_accs else 1.0
    print(f'[{args.variant}] B=1 speedup: baseline={base_tps:.1f} tok/s | DSpark={ds_tps:.1f} tok/s ({speedup_b1:.2f}x) | tau_b1={tau_b1:.3f}', flush=True)
    del eng_sp

    # 2) Measure B=64 pass@k & tau across all 100 prompts (same seeds/u0 as results_maze.json!)
    t0_all = time.perf_counter()
    hits_all = []
    all_accs = []
    total_gen_toks = 0
    curves = {k: [] for k in K_VALUES}
    for i, it in enumerate(prompts):
        gen = torch.Generator(device=dev).manual_seed(SEED * 1000 + i)
        u0 = lattice(B, gen, dev)
        eng = S5VariantEngine(model, drafter, tok, temperature=T, seed=SEED * 1000 + i, draft_variant=args.variant)
        chains = eng.generate_branches(it['ids'], max_new, n=ND, branches=B,
                                       mode='lattice', return_all=True, u0=u0)
        hits = [int(check_maze(tok.decode(x, skip_special_tokens=True), it['maze_grid']))
                for x, _ in chains]
        for x, st in chains:
            total_gen_toks += len(x)
            all_accs.extend(st['accepted'])
        hits_all.append(hits)
        for k in K_VALUES:
            curves[k].append(lattice_est(hits, k))
        del eng
        if (i + 1) % 20 == 0 or i == 0:
            print(f'  [{args.variant}] prompt {i+1}/{len(prompts)} hits={sum(hits)}/{B}', flush=True)

    wall_b64 = time.perf_counter() - t0_all
    mean_tau_b64 = (sum(all_accs) / len(all_accs) + 1.0) if all_accs else 1.0
    tps_b64 = total_gen_toks / wall_b64

    out_data = {
        'variant': args.variant,
        'curve': {str(k): round(sum(v) / len(v), 4) for k, v in curves.items()},
        'hits': hits_all,
        'mean_tau_b64': round(mean_tau_b64, 3),
        'tps_b64': round(tps_b64, 1),
        'wall_sec_b64': round(wall_b64, 1),
        'baseline_b1_tps': round(base_tps, 1),
        'dspark_b1_tps': round(ds_tps, 1),
        'speedup_b1': round(speedup_b1, 3),
        'tau_b1': round(tau_b1, 3),
    }
    json.dump(out_data, open(args.out, 'w'), indent=2)
    print(f'[{args.variant}] DONE: ' + ' '.join(f'pass@{k}={out_data["curve"][str(k)]*100:.1f}%' for k in K_VALUES)
          + f' | tau_b64={mean_tau_b64:.3f} | B64_tps={tps_b64:.1f} tok/s ({wall_b64:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
