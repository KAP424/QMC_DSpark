'''Correctness harness for dspark_fast.DSparkDecodeFast (multi-branch batched-verify engine).

Test A (rewind unit): forced rejection, fast path (GDN restore + delta-rule rewind + 1-token corr
forward) vs reference path (deepcopy restore + replay forward) — next-token logits at the corr.
Test B (losslessness, decisive): at every round the engine's verify logits vs a from-scratch
recomputation (fresh prefill + ONE forward over prompt + committed + chunk).
Test B2: the same harness on the OLD engine — the noise-floor reference.
Test C: T=0.05 near-greedy token agreement old vs fast.
Test D: speed + branches sweep (B independent chains verified in one batched forward).
'''
import sys
import time

import torch

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_paper import DSparkDrafter, ND, load as dload  # noqa: E402
from dspark_e2e import _pos  # noqa: E402
import dspark_fast  # noqa: E402

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
CK = '/home/dev/KAP/parameter/draft/dspark_pb_v2_b/best.pt'
DEV = 'cuda:0'
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
d = DSparkDrafter(model, n_draft=ND, layers=5, markov_rank=128, device=DEV, dtype=torch.float32)
ck = torch.load(CK, map_location=DEV, weights_only=False)
d.load_state_dict(ck['model'])
d.eval()
print(f'drafter {CK} (tau {ck["tau"]:.3f})')


def ids_of(p):
    return tok(tok.apply_chat_template([{'role': 'user', 'content': p}],
                                       tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)['input_ids']


PROMPTS = [ids_of('Explain why the sky is blue, then list three colors.'),
           ids_of('Write a short Python function that reverses a string.'),
           ids_of('The quick brown fox jumps over the lazy dog while autumn rain falls on the quiet village roofs.')]


def reference_logits(prompt_ids, committed, chunk):
    full = torch.tensor([list(prompt_ids) + list(committed) + list(chunk)], device=DEV)
    with torch.inference_mode():
        o = model(input_ids=full, use_cache=False, logits_to_keep=len(chunk) + 1)
    return o.logits[0].float()


# ---------- Test A: rewind vs replay ----------
print('\n[Test A] rewind-vs-replay, forced rejection at k=2')
eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=0)
with torch.inference_mode():
    ids = torch.tensor([PROMPTS[0]], device=DEV)
    cache, last, feats = eng.prefill(ids)
    pos = ids.shape[1]
    anchor = eng._sample(last)
    q_lg, dr, _ = eng._propose_multi(feats[0], anchor, pos, ND, 1)
    import copy as _copy
    cache_pre = _copy.deepcopy(cache)
    snap = eng.rw.snap(cache_pre)
    chunk = torch.cat([anchor.view(1, 1), dr], 1)                                   # (1,ND+1)
    v_lg, v_feats, hs, cache = eng._forward(cache, chunk, pos)
    v_lg_r, v_feats_r, hs_r, cache_r = eng._forward(_copy.deepcopy(cache_pre), chunk, pos)
    k = 2
    corr = torch.tensor([1234], device=DEV)
    prefix = chunk[0, :k + 1]
    o_ref = model(input_ids=torch.cat([prefix, corr.view(1)]).view(1, -1),
                  past_key_values=_copy.deepcopy(cache_pre),
                  position_ids=dspark_fast._pos2(pos, k + 2, 1, DEV), use_cache=True, logits_to_keep=1)
    lg_ref = o_ref.logits[0, -1].float()
    eng.rw.restore(cache, snap, n_crop=(ND + 1) - (k + 1))
    eng.rw.rewind(cache, snap, hs, k + 1, bidx=0)
    o_fast = model(input_ids=corr.view(1, 1), past_key_values=cache,
                   position_ids=dspark_fast._pos2(pos + k + 1, 1, 1, DEV), use_cache=True, logits_to_keep=1)
    lg_fast = o_fast.logits[0, -1].float()
    diff = (lg_ref - lg_fast).abs()
print(f'  next-token logits after corr: max|diff| {diff.max():.4f}  mean {diff.mean():.5f}  '
      f'argmax equal {bool(lg_ref.argmax() == lg_fast.argmax())}')
pd, pf = torch.softmax(v_lg[0, k], -1), torch.softmax(v_lg_r[0, k], -1)
print(f'  verify-path self-check (twin caches): max|p-q| {(pd - pf).abs().max():.2e}')

# ---------- Test B: per-round logit parity (fast engine) ----------
print('\n[Test B] per-round logit parity vs from-scratch recomputation (losslessness)')
for pi, pr in enumerate(PROMPTS):
    eng = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=100 + pi)
    with torch.inference_mode():
        ids = torch.tensor([pr], device=DEV)
        cache, last, feats = eng.prefill(ids)
        pos = ids.shape[1]
        anchor = eng._sample(last)
        committed, diffs, rounds = [], [], 0
        while rounds < 15 and len(committed) < 120:
            q_lg, dr, _ = eng._propose_multi(feats[0], anchor, pos, ND, 1)
            snap = eng.rw.snap(cache)
            chunk = torch.cat([anchor.view(1, 1), dr], 1)
            v_lg, v_feats, hs, cache = eng._forward(cache, chunk, pos)
            ref = reference_logits(pr, committed, chunk[0].tolist())
            diffs.append((v_lg[0] - ref[1:]).abs().max().item())
            k, corr, bidx = eng._accept_multi(q_lg, dr, v_lg[:, :ND])
            rounds += 1
            if corr is None:
                committed += chunk[0].tolist()
                feats = torch.cat([feats, v_feats[bidx:bidx + 1]], 1)
                pos += ND + 1
                anchor = eng._sample(v_lg[bidx, ND])
            else:
                committed += chunk[bidx, :k + 1].tolist()
                eng.rw.restore(cache, snap, n_crop=(ND + 1) - (k + 1))
                eng.rw.rewind(cache, snap, hs, k + 1, bidx=bidx)
                feats = torch.cat([feats, v_feats[bidx:bidx + 1, :k + 1]], 1)
                pos += k + 1
                anchor = corr
            if eng.tok.eos_token_id in committed or int(anchor) == eng.tok.eos_token_id:
                break
        print(f'  prompt {pi}: rounds {rounds}  tokens {len(committed)}  '
              f'max|v_lg-ref| = {max(diffs):.4f}  mean {sum(diffs) / len(diffs):.4f}  '
              f'text: {tok.decode(committed[:36])!r}'[:150])

# ---------- Test B2: old-engine parity (noise-floor reference) ----------
print('\n[Test B2] old-engine parity vs the same from-scratch recomputation')
from dspark_e2e import DSparkDecode  # noqa: E402
import copy as _copy  # noqa: E402
for pi, pr in enumerate(PROMPTS[:2]):
    eng = DSparkDecode(model, d, tok, temperature=1.0, seed=100 + pi)
    with torch.inference_mode():
        ids = torch.tensor([pr], device=DEV)
        cache, last, feats = eng.prefill(ids)
        pos = ids.shape[1]
        anchor_v = eng._sample(last)
        committed, diffs, rounds = [], [], 0
        while rounds < 15 and len(committed) < 120:
            anchor = int(anchor_v)
            q_lg, dr = eng.propose(feats, torch.tensor([anchor], device=DEV), pos)
            chunk = [anchor] + dr.tolist()
            snap = _copy.deepcopy(cache)
            v_lg, v_feats, cache = eng._forward(cache, torch.tensor(chunk, device=DEV), pos)
            ref = reference_logits(pr, committed, chunk)
            diffs.append((v_lg[:ND] - ref[1:ND + 1]).abs().max().item())
            k, corr = eng._accept(q_lg, dr, v_lg[:ND])
            rounds += 1
            if corr is None:
                committed += chunk
                feats = torch.cat([feats, v_feats], 0)
                pos += ND + 1
                anchor_v = eng._sample(v_lg[ND])
            else:
                new = chunk[:k + 1] + [int(corr)]
                r_lg, r_feats, cache = eng._forward(snap, torch.tensor(new, device=DEV), pos)
                committed += new
                feats = torch.cat([feats, r_feats], 0)
                pos += len(new)
                anchor_v = eng._sample(r_lg[-1])
            if eng.tok.eos_token_id in committed:
                break
        print(f'  old-engine prompt {pi}: rounds {rounds}  max|v_lg-ref| = {max(diffs):.4f}  '
              f'mean {sum(diffs) / len(diffs):.4f}')

# ---------- Test D: speed + branches sweep ----------
print('\n[Test D] speed / branches sweep, T=1.0, 2 prompts x 200 tokens')
for pi, pr in enumerate(PROMPTS[:2]):
    for B in (1, 2, 4, 8):
        fast = dspark_fast.DSparkDecodeFast(model, d, tok, temperature=1.0, seed=42 + pi)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, st = fast.generate(pr, 200, branches=B)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'  prompt {pi} B={B}: {st["tokens"] / dt:6.1f} tok/s  tau {st["tokens"] / st["rounds"]:5.2f}  '
              f'rounds {st["rounds"]:3d}  all-acc {st["all_accepted"] / st["rounds"]:.2f}  '
              f'draft {st["draft_ms"] / st["rounds"]:4.1f} verify {st["verify_ms"] / st["rounds"]:4.1f} '
              f'rewind {st["rewind_ms"] / st["rounds"]:4.1f} ms')
print('\ndone')
