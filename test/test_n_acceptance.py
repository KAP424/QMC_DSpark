"""测试 draft token 数目（n）对平均接受 token 数的影响。
使用 generate_branches(B=4, return_all=True)，T=1.0，每个 n 值测 2 个 prompt x 4 条链，
统计：每条链的 mean accepted/round，以及最终 tok/s。"""
import sys, time, statistics
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
print(f'drafter tau {ck.get("tau"):.3f}  ND_max={ND}')

def ids_of(p):
    return tok(tok.apply_chat_template([{'role': 'user', 'content': p}],
                                       tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)['input_ids']

PROMPTS = [ids_of('Explain why the sky is blue, then list three colors.'),
           ids_of('Write a short Python function that reverses a string.')]

print(f'\n{"n":>4}  {"mean_k":>8}  {"mean_tau":>9}  {"tok/s":>8}  {"E[k]/n":>7}')
print('-' * 50)
for n in range(1, ND + 1):
    all_k, all_tau, spd = [], [], []
    for pi, pr in enumerate(PROMPTS):
        eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        results = eng.generate_branches(pr, 128, n=n, branches=4,
                                        mode='stratified', return_all=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        for out_ids, st in results:
            if st['rounds'] > 0:
                all_k.extend(st['accepted'])
                all_tau.append(st['tokens'] / st['rounds'])
                spd.append(st['tokens'] / dt)
    mk = statistics.mean(all_k) if all_k else 0
    mt = statistics.mean(all_tau) if all_tau else 0
    ms = statistics.mean(spd) if spd else 0
    print(f'{n:>4}  {mk:>8.3f}  {mt:>9.3f}  {ms:>8.1f}  {mk/n:>7.3f}')

print('\n説明:')
print('  mean_k    : 每轮平均接受的草稿 token 数 E[k]')
print('  mean_tau  : 每轮平均提交 token 数 = E[k] + 1（含 anchor/correction）')
print('  E[k]/n    : 接受率（越高说明 drafter 在当前 n 下匹配越好）')
print('  tok/s     : 4 条链总 token 除以总耗时（包含 B=4 条链的串行开销）')
