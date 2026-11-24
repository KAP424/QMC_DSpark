'''Layer-by-layer debug of the GDN rewind: find the first divergence vs a true 4-token forward.'''
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/dev/KAP/src4dspark')
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from dspark_e2e import _pos  # noqa: E402
import dspark_fast  # noqa: E402

P = '/home/dev/KAP/parameter/target/Qwen3.5-0.8B-Base-original'
DEV = 'cuda:0'
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map=DEV).eval()
rw = dspark_fast.GdnRewinder(model)
core = model.model
lm = getattr(core, 'language_model', None) or getattr(core, 'model', core)
layers = lm.layers

prompt = tok('The quick brown fox jumps over the lazy dog and the autumn rain falls softly.',
             return_tensors='pt')['input_ids'].to(DEV)
with torch.inference_mode():
    o = model(input_ids=prompt, use_cache=True, output_hidden_states=True, logits_to_keep=1)
cache = o.past_key_values
S = prompt.shape[1]
snap = rw.snap(cache)
hs_true = o.hidden_states

new_ids = torch.tensor([[100, 200, 300, 400]], device=DEV)
with torch.inference_mode():
    o2 = model(input_ids=new_ids, past_key_values=cache, position_ids=_pos(S, 4, DEV),
               use_cache=True, output_hidden_states=True, logits_to_keep=1)
hs_new = o2.hidden_states                       # entry i = input of decoder layer i, (1,4,1024)

# my rewind into a fresh copy of the pre-forward states
import copy as _copy
cache2 = _copy.deepcopy(cache)                   # state BEFORE the 4 tokens
snap2 = rw.snap(cache2)
with torch.inference_mode():
    rw.rewind(cache2, snap2, hs_new, 4)

print('per-layer |recurrent_rewind - recurrent_true| and |conv_rewind - conv_true|:')
worst_r = worst_c = 0.0
for b, i in enumerate(rw.gdn_layer_idx):
    lt = cache.layers[i]
    l2 = cache2.layers[i]
    dr = (lt.recurrent_states[0] - l2.recurrent_states[0]).abs().max().item()
    dc = (lt.conv_states[0] - l2.conv_states[0]).abs().max().item()
    worst_r, worst_c = max(worst_r, dr), max(worst_c, dc)
    if dr > 0.05 or dc > 0.05 or b < 2:
        print(f'  layer {i:2d}: d recurrent {dr:.4f}   d conv {dc:.5f}')
print(f'worst: recurrent {worst_r:.4f}, conv {worst_c:.5f}')

# now dissect layer by layer on the FIRST gdn layer: mixed, conv-out, q/k/v
i0 = rw.gdn_layer_idx[0]
mod = layers[i0].linear_attn
h_true = layers[i0].input_layernorm(hs_new[i0][0])                 # (4,1024)
with torch.inference_mode():
    mixed_true = mod.in_proj_qkv(h_true).transpose(1, 2)           # (1,6144,4)
    full_true = cache.layers[i0].conv_states[0] if False else None
# recompute the layer's conv exactly as its code does, from the PRE-4-token state:
conv_pre = cache2.layers[i0].conv_states[0]                        # snapshot state (1,6144,4)
h_mine = (hs_new[i0][0].float() * torch.rsqrt(hs_new[i0][0].float().pow(2).mean(-1, keepdim=True) + rw.eps))
h_mine = (h_mine * rw.ln_w[0]).to(torch.bfloat16)
mixed_mine = torch.bmm(h_mine.unsqueeze(0), rw.w_qkv[0].t().unsqueeze(0)).transpose(1, 2)
print('\nlayer', i0, '| mixed diff:', (mixed_true - mixed_mine).abs().max().item())
from causal_conv1d import causal_conv1d_fn  # noqa: E402
full = torch.cat([conv_pre, mixed_true], -1)
y_true = causal_conv1d_fn(full, mod.conv1d.weight.squeeze(1), None, activation='silu')[:, :, -4:]
acc = torch.zeros(1, rw.conv_dim, 4, device=DEV)
fc_mine = torch.cat([conv_pre, mixed_mine], -1)
for m_ in range(4):
    acc = acc + rw.conv_w[0][:, m_].view(1, -1, 1) * fc_mine[:, :, m_ + 1:m_ + 5]
y_mine = F.silu(acc)
print('conv-out diff (true vs mine):', (y_true - y_mine).abs().max().item())
q_t, k_t, v_t = torch.split(y_true.transpose(1, 2), [2048, 2048, 2048], -1)
q_m, k_m, v_m = torch.split(y_mine.transpose(1, 2), [2048, 2048, 2048], -1)
print('q diff:', (q_t - q_m).abs().max().item(), ' k diff:', (k_t - k_m).abs().max().item(),
      ' v diff:', (v_t - v_m).abs().max().item())
