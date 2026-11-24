'''测试 generate_branches：B 条链全部保留，每条用自己的残差 uniform 继续单链解码。'''
import sys, time
import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer
from dspark_paper import DSparkDrafter, ND
import dspark_fast

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
CK = '/home/dev/KAP/parameter/draft/dspark_domain/best.pt'
DEV = 'cuda:0'

tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CK, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()
print(f'drafter tau {ck.get("tau"):.3f}')

def ids_of(p):
    return tok(tok.apply_chat_template([{'role': 'user', 'content': p}],
                                       tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)['input_ids']

PROMPTS = [ids_of('Explain why the sky is blue, then list three colors.'),
           ids_of('Write a short Python function that reverses a string.')]

# ---- Test 1: 基本运行，全部 B 条链都产出 ----
print('\n[Test 1] generate_branches: B=4, 返回全部链')
for pi, pr in enumerate(PROMPTS):
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42 + pi)
    t0 = time.perf_counter(); torch.cuda.synchronize()
    results = eng.generate_branches(pr, 128, branches=4, mode='stratified', return_all=True)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f'\n  prompt {pi} ({dt:.1f}s total for {len(results)} chains):')
    for bi, (out, st) in enumerate(results):
        print(f'    chain {bi}: {st["tokens"]} tok  rounds {st["rounds"]}  '
              f'tau {st["tokens"]/st["rounds"]:.2f}  '
              f'text: {tok.decode(out[:30])!r}'[:100])

# ---- Test 2: 链间多样性 (T=0.7) ----
print('\n[Test 2] 链间多样性：每条链前 20 token')
pr = PROMPTS[1]
eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=0.7, seed=0)
results = eng.generate_branches(pr, 60, branches=4, mode='stratified', return_all=True)
for bi, (out, _) in enumerate(results):
    print(f'  chain {bi}: {tok.decode(out[:20])!r}')

# ---- Test 3: 与 generate(B=4) 速度对比 ----
print('\n[Test 3] 速度对比 (T=1.0, 2 prompts x 128 tok)')
for pi, pr in enumerate(PROMPTS):
    for name, fn_kwargs in (
        ('generate  B=4', ('generate',  dict(branches=4, mode='stratified'))),
        ('branches  B=4', ('generate_branches', dict(branches=4, mode='stratified', return_all=False))),
    ):
        eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=7 + pi)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn = getattr(eng, fn_kwargs[0])
        out, st = fn(pr, 128, **fn_kwargs[1])
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f'  prompt {pi} {name}: {st["tokens"]/dt:6.1f} tok/s  '
              f'tau {st["tokens"]/st["rounds"]:.2f}  rounds {st["rounds"]}  '
              f'forwards {st["forwards"]}')
print('\ndone')
