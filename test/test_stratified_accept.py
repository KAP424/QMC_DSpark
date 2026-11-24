'''分层逆 CDF 采样 vs i.i.d. Gumbel：接受率会不会下降？

Part 1（边际精确性，决定性）：在真实 prompt 状态上，把两种采样器的
  - 槽 1 token 频率 vs 解析 q1
  - 槽 2 条件频率 E[x2 | x1=c*] vs 解析 q2(.|c*)
逐一对比 TV 距离。若 stratified 的偏差与 gumbel 同为采样噪声级别，则提案边际无偏 => 接受率不可能下降。

Part 2（端到端实测）：12 prompt x 256 token，B=4，两种模式各跑一遍，
对比每轮接受草稿数 mean(k)、tau、P(k>=m) 曲线。
'''
import sys, time
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
sys.path.insert(0, '/home/dev/KAP/train')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
import dspark_fast

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
W = '/home/dev/KAP/data/pb_dspark'
DEV = 'cuda:0'
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load('/home/dev/KAP/parameter/draft/dspark_pb_v2_b/best.pt', map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()

items = [c for c in torch.load(W + '/prompts.pt') if c['split'] == 'val']
prompts = [c['ids'] for c in items if 60 <= len(c['ids']) <= 400][:12]

# ---------- Part 1: marginal exactness ----------
print('[Part 1] 提案分布边际精确性（stratified vs gumbel vs 解析）')
prompt = prompts[0]
eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=0)
with torch.inference_mode():
    ids = torch.tensor([prompt], device=DEV)
    cache, last, feats = eng.prefill(ids)
    pos = ids.shape[1]
    anchor = eng._sample(last)
    # base logits（解析参考的来源）
    hctx = d.encode_ctx(feats[0, :pos].unsqueeze(0))
    o = torch.tensor([[pos - 1]], device=DEV)
    z = d.backbone(hctx, d.embed(anchor.view(1, 1)), o)[0, 0]
    base = d.lm_head(z.to(d.lm_head.weight.dtype)).float()          # (7, V)
    q1_exact = torch.softmax(base[0], -1)

    M1 = 200_000
    # 两种 uniform：stratified（8 层结构，重复 reps 组）vs 完全独立
    Bp = 8
    reps = M1 // Bp
    v = torch.rand(reps, Bp, device=DEV)
    u_strat = ((torch.arange(Bp, device=DEV).float() + v) / Bp).reshape(-1)             # (M1,)
    u_indep = torch.rand(M1, device=DEV)
    cdf1 = q1_exact.cumsum(-1)
    x1_s = torch.searchsorted(cdf1, u_strat, right=True).clamp(max=len(q1_exact) - 1)
    x1_g = torch.searchsorted(cdf1, u_indep, right=True).clamp(max=len(q1_exact) - 1)
    # 注：argmax(log q + Gumbel) 与 searchsorted(cdf, u) 是同一分布的两种实现，这里用后者以便 20 万次批量

    def freq(x):
        return torch.bincount(x, minlength=len(q1_exact)).float() / len(x)
    def tv(a, b):
        return 0.5 * (a - b).abs().sum().item()

    f_s, f_g = freq(x1_s), freq(x1_g)
    print(f'  slot-1: M={M1}')
    print(f'    TV(stratified, exact) = {tv(f_s, q1_exact):.5f}')
    print(f'    TV(indep-uniform, exact) = {tv(f_g, q1_exact):.5f}   <- 两者应同为采样噪声级别')

    # slot-2 条件分布 E[x2 | x1=c*]：条件相同时所有样本共享同一 q2/cdf2，只需比较残差 u 的去向
    c_star = int(q1_exact.argmax())
    M2 = 100_000
    q2_exact = torch.softmax((base[1] + d.markov(torch.tensor([c_star], device=DEV))[0]) / 1.0, -1)
    cdf2 = q2_exact.cumsum(-1)
    def slot2_from_u1(u1):                     # 给定槽1 的 u，走链到槽 2
        x1 = torch.searchsorted(cdf1, u1, right=True).clamp(max=len(q1_exact) - 1)
        qx1 = q1_exact.gather(0, x1)
        lo = cdf1.gather(0, x1) - qx1
        ur = ((u1 - lo) / qx1).clamp(0, 1 - 1e-7)          # 嵌套残差（stratified 的耦合处）
        x2 = torch.searchsorted(cdf2, ur, right=True).clamp(max=len(q2_exact) - 1)
        return x1, x2
    # stratified：分层结构走链，保留 x1=c* 的样本
    v2 = torch.rand(M2 // Bp, Bp, device=DEV)
    u1_s = ((torch.arange(Bp, device=DEV).float() + v2) / Bp).reshape(-1)
    x1_s2, x2_s = slot2_from_u1(u1_s)
    keep_s = x1_s2 == c_star
    # 独立 uniform：同样走链（= gumbel/逆 CDF 的边际形式）
    u1_g = torch.rand(M2, device=DEV)
    x1_g2, x2_g = slot2_from_u1(u1_g)
    keep_g = x1_g2 == c_star
    print(f'  slot-2 条件分布 E[x2 | x1={c_star}]（stratified n={int(keep_s.sum())}, indep n={int(keep_g.sum())}）')
    print(f'    TV(stratified, exact) = {tv(freq(x2_s[keep_s]), q2_exact):.5f}')
    print(f'    TV(indep-uniform, exact) = {tv(freq(x2_g[keep_g]), q2_exact):.5f}')

# ---------- Part 2: 端到端接受率 ----------
print('\n[Part 2] 端到端接受率（12 prompt x 256 token, B=4, T=1.0）')
for mode in ('iid', 'stratified'):
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=7)
    ks, taus, tokss = [], [], []
    for p in prompts:
        out, st = eng.generate(p, 256, branches=4, mode=mode)
        ks += st['accepted']
        taus.append(st['tokens'] / st['rounds'])
    ks_t = torch.tensor(ks, dtype=torch.float32)
    curve = [float((ks_t >= m).float().mean()) for m in range(1, ND + 1)]
    print(f'  {mode:10s}: mean accepted/round {ks_t.mean():.3f}  tau {sum(taus)/len(taus):.3f}  '
          f'P(k>=m) = ' + ' '.join(f'{c:.2f}' for c in curve))
print('\n说明：P(k>=m) 是"每轮至少接受 m 个草稿"的频率；两条曲线一致 => 接受率无差异。')
