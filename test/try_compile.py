'''Tier-3 experiment: torch.compile the target text stack and measure the 8-token cached forward.'''
import copy
import statistics
import sys
import time

import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_e2e import _pos  # noqa: E402

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV = 'cuda:0'
SYNC = torch.cuda.synchronize
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
SYNC()


def make_cache(prompt_ids, target_S=200):
    grow = max(0, target_S - len(prompt_ids))
    with torch.inference_mode():
        o = model(input_ids=torch.tensor([prompt_ids], device=DEV), use_cache=True, logits_to_keep=1)
        cache = o.past_key_values
        if grow:
            model(input_ids=torch.randint(1000, 20000, (1, grow), device=DEV), past_key_values=cache,
                  position_ids=_pos(len(prompt_ids), grow, DEV), use_cache=True, logits_to_keep=1)
    return cache, cache.layers[3].get_seq_length()


prompt = tok('The quick brown fox jumps over the lazy dog while the autumn rain falls softly on the roofs.',
             return_tensors='pt')['input_ids'][0].tolist()
cache0, S = make_cache(prompt)
ids8 = torch.randint(1000, 20000, (1, 8), device=DEV)


def timed(fn, n=10):
    ts = []
    for _ in range(n):
        c = copy.deepcopy(cache0)
        SYNC(); t0 = time.perf_counter()
        fn(c)
        SYNC(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def fwd8(c, **kw):
    with torch.inference_mode():
        return model(input_ids=ids8, past_key_values=c, position_ids=_pos(S, 8, DEV), use_cache=True,
                     logits_to_keep=8, **kw)


base_out = fwd8(copy.deepcopy(cache0), output_hidden_states=True).logits[0].float()
print(f'eager 8-token cached forward: {timed(fwd8):.2f} ms')

lm = model.model.language_model
t0 = time.perf_counter()
model.model.language_model = torch.compile(lm)
c = copy.deepcopy(cache0)
fwd8(c)  # trigger compilation
SYNC()
print(f'compile time: {time.perf_counter() - t0:.0f} s')
for _ in range(2):
    fwd8(copy.deepcopy(cache0))
SYNC()
print(f'compiled 8-token cached forward: {timed(fwd8):.2f} ms')
out_c = fwd8(copy.deepcopy(cache0)).logits[0].float()
print(f'logits parity eager-vs-compiled: max|diff| {(base_out - out_c).abs().max().item():.4f}')
