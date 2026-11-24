'''测试 generate_tree：正确性 + 基本行为 + 速度。'''
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
for p in model.parameters():
    p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CK, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model']); d.eval()
print(f'drafter {CK} (tau {ck.get("tau"):.3f})')

def ids_of(p):
    return tok(tok.apply_chat_template([{'role': 'user', 'content': p}],
                                       tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)['input_ids']

PROMPTS = [ids_of('Explain why the sky is blue, then list three colors.'),
           ids_of('Write a short Python function that reverses a string.')]

# ---- Test 1: 基本运行 + 输出长度 ----
print('\n[Test 1] generate_tree 基本运行')
for pi, pr in enumerate(PROMPTS):
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42 + pi)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out, st = eng.generate_tree(pr, 128, n=ND, branches=4, mode='stratified')
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f'  prompt {pi}: {st["tokens"]} tok in {dt:.1f}s = {st["tokens"]/dt:.1f} tok/s  '
          f'rounds {st["rounds"]}  forwards {st["forwards"]}  pruned {st["pruned"]}')
    print(f'    text: {tok.decode(out[:50])!r}'[:140])

# ---- Test 2: 无损性（vs HF generate, T=0.05 近 greedy）----
print('\n[Test 2] T=0.05 近 greedy, vs HF generate 的 token 一致性')
for pi, pr in enumerate(PROMPTS):
    t = torch.tensor([pr], device=DEV)
    torch.manual_seed(100 + pi)
    with torch.inference_mode():
        hfo = model.generate(input_ids=t, max_new_tokens=64, do_sample=True,
                             temperature=0.05, top_p=1.0, pad_token_id=tok.eos_token_id)
    hf_out = hfo[0][t.shape[1]:].tolist()
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=0.05, seed=100 + pi)
    tree_out, st = eng.generate_tree(pr, 64, n=ND, branches=2, mode='stratified')
    m = sum(a == b for a, b in zip(hf_out, tree_out))
    print(f'  prompt {pi}: {m}/{min(len(hf_out), len(tree_out))} 一致 (hf {len(hf_out)}, tree {len(tree_out)})  '
          f'tree rounds {st["rounds"]}, pruned {st["pruned"]}')

# ---- Test 3: branches 扫描（B=2/4/8）----
print('\n[Test 3] branches 扫描 (T=1.0, 2 prompts x 200 tok)')
for pi, pr in enumerate(PROMPTS):
    for B in (2, 4, 8):
        eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=7 + pi)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, st = eng.generate_tree(pr, 200, n=ND, branches=B, mode='stratified')
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f'  prompt {pi} B={B}: {st["tokens"]/dt:6.1f} tok/s  rounds {st["rounds"]:3d}  '
              f'forwards {st["forwards"]}  pruned {st["pruned"]}  '
              f'mean_acc {sum(st["accepted"])/len(st["accepted"]):.2f}')
print('\ndone')
