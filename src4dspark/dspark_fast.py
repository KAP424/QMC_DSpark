"""Fast DSpark engine: one batched target forward per round, GDN rollback by delta-rule rewind,
and optional multi-branch rollout verification.

vs dspark_e2e.DSparkDecode ("deepcopy snapshot + replay forward", 2 forwards/round):

  1. Snapshot = the 18 GDN (conv, recurrent) state clones only (~0.2 ms); the 6 attention layers'
     KV is append-only per-token and simply cropped back on rejection.
  2. On rejection the GDN recurrent state after the accepted prefix is RECONSTRUCTED
     mathematically: the gated delta rule is prefix-associative, so
         S(after prefix) = RecurStep(S_before_round, factors of accepted tokens)
     with one batched fla `fused_recurrent_gated_delta_rule` call over all 18 layers. The per-token
     factors (post-conv q/k/v, beta, g) are recomputed OUTSIDE the layer from the verify forward's
     own hidden_states. Verified against a true forward: max abs diff 2.6e-3 (bf16 noise).
  3. The correction token becomes the next round's anchor (it is a lawful target sample), so the
     verify chunk is always [anchor, d1..dn] — no separate decode/replay forward, ever.
  4. Arithmetic-coding semantics (QuasiMoTTo): each beam owns ONE sequence uniform u_b;
     the committed sequence is exactly Phi_p(u_b), the target's inverse-CDF decode of u_b.
     At every slot the committed token is x*_t = F_p^{-1}(u_t); the draft is accepted iff
     it equals x*_t (otherwise x*_t is committed and the block stops), and u_t updates as
     the residual under p. The drafter therefore never changes the output distribution —
     it only decides how many target forwards are skipped. Round 1 couples the B uniforms
     with a random-permuted stratification (QMC) or iid draws; afterwards each beam evolves
     its own uniform deterministically. NOTE: the acceptance rate under this rule is
     P(draft == target CDF token), lower than the classical min(1,p/q) — that is the price
     of making the whole sequence a deterministic function of one uniform.
  5. `branches=B`: sample B independent draft chains (shared drafter base logits; batched markov),
     verify them as a batch of B sequences in the SAME target forward, commit the deepest-accepted
     chain (+ its correction). On this launch-bound engine the batched verify costs the same wall
     time as B=1 (measured 30.1/29.8/30.3/30.3 ms at B=1/2/4/8) — extra rollouts are ~free.
     GDN layers cannot share state across tree branches, so branches are isolated on the batch
     dimension (each gets its own copy of the prefix state), NOT via tree attention.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dflash_big import CTX_HS  # noqa: E402
from dspark_paper import ND  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule  # noqa: E402


def _pos2(pos0, length, B, dev):
    return torch.arange(pos0, pos0 + length, device=dev).view(1, -1).expand(B, -1).contiguous()


class GdnRewinder:
    """Snapshot / restore / mathematically rewind the 18 GDN layers of a Qwen3.5 cache."""

    def __init__(self, model):
        cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
        types = cfg.layer_types
        self.dev = model.device
        self.gdn_layer_idx = [i for i, t in enumerate(types) if t == 'linear_attention']
        assert len(self.gdn_layer_idx) == 18
        core = model.model
        lm = getattr(core, 'language_model', None) or getattr(core, 'model', core)
        layers = lm.layers
        mods = [layers[i].linear_attn for i in self.gdn_layer_idx]
        lns = [layers[i].input_layernorm for i in self.gdn_layer_idx]
        self.ln_w = torch.stack([(1.0 + ln.weight.float()) for ln in lns]).to(self.dev)      # (18,1024)
        self.eps = lns[0].eps
        self.w_qkv = torch.stack([m.in_proj_qkv.weight for m in mods]).to(self.dev)          # (18,6144,1024)
        self.w_b = torch.stack([m.in_proj_b.weight for m in mods]).to(self.dev)              # (18,16,1024)
        self.w_a = torch.stack([m.in_proj_a.weight for m in mods]).to(self.dev)              # (18,16,1024)
        self.conv_w = torch.stack([m.conv1d.weight.squeeze(1) for m in mods]).to(self.dev)   # (18,6144,4)
        self.a_log = torch.stack([m.A_log for m in mods]).to(self.dev)                       # (18,16)
        self.dt_bias = torch.stack([m.dt_bias for m in mods]).to(self.dev)                   # (18,16)
        assert cfg.hidden_act == 'silu'
        self.kern = cfg.linear_conv_kernel_dim
        self.kdim, self.vdim = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        assert self.nk == self.nv
        self.conv_dim = self.nk * self.kdim * 2 + self.nv * self.vdim

    def snap(self, cache):
        out = []
        for i in self.gdn_layer_idx:
            l = cache.layers[i]
            out.append((l.conv_states[0].clone(), l.recurrent_states[0].clone()))
        return out

    def restore(self, cache, snap, n_crop):
        """GDN states <- snapshot; attention KV <- remove the last n_crop rows."""
        for b, i in enumerate(self.gdn_layer_idx):
            l = cache.layers[i]
            l.conv_states[0].copy_(snap[b][0])
            l.recurrent_states[0].copy_(snap[b][1])
        if n_crop:
            for i, l in enumerate(cache.layers):
                if i not in self.gdn_layer_idx:
                    l.crop(-n_crop)

    def rewind(self, cache, snap, hs, j, bidx=0):
        """Rebuild GDN states as of chunk position j-1 of verify row `bidx`.

        hs: hidden_states tuple from the (possibly batched) verify forward; entry i = input of
        decoder layer i, (B, L, H). Writes conv+recurrent states of all 18 GDN layers in place.
        """
        h = torch.stack([hs[i][bidx, :j] for i in self.gdn_layer_idx], 0)                   # (18,j,1024)
        h = (h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + self.eps))
        h = (h * self.ln_w[:, None, :]).to(self.w_qkv.dtype)
        mixed = torch.bmm(h, self.w_qkv.transpose(1, 2))                                    # (18,j,6144)
        beta = torch.sigmoid(torch.bmm(h, self.w_b.transpose(1, 2)))                        # (18,j,16)
        a = torch.bmm(h, self.w_a.transpose(1, 2))
        g = -self.a_log.float().exp()[:, None, :] * F.softplus(a.float() + self.dt_bias.float()[:, None, :])
        full_conv = torch.cat([torch.stack([s[0][0] for s in snap], 0),
                               mixed.transpose(1, 2)], -1)                                  # (18,6144,4+j)
        acc = torch.zeros_like(full_conv[:, :, :j])
        for m_ in range(self.kern):
            acc = acc + self.conv_w[:, :, m_].unsqueeze(-1) * full_conv[:, :, m_ + 1:m_ + 1 + j]
        y = F.silu(acc)
        qkv = y.transpose(1, 2)
        q, k, v = torch.split(qkv, [self.nk * self.kdim, self.nk * self.kdim, self.nv * self.vdim], -1)
        q = q.view(18, j, self.nk, self.kdim)
        k = k.view(18, j, self.nk, self.kdim)
        v = v.view(18, j, self.nv, self.vdim)
        _, final = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=torch.stack([s[1][0] for s in snap], 0),
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        for b, i in enumerate(self.gdn_layer_idx):
            l = cache.layers[i]
            l.recurrent_states[0].copy_(final[b])
            l.conv_states[0].copy_(full_conv[b][:, -self.kern:])


def _is_attn_layer(l):
    return not hasattr(l, 'recurrent_states')


class DSparkDecodeFast:
    def __init__(self, model, drafter, tokenizer, temperature=1.0, seed=0):
        self.m, self.d, self.tok = model, drafter, tokenizer
        self.T = temperature
        self.gen = torch.Generator(device=model.device).manual_seed(seed)
        self.dev = model.device
        # torch.compile note (test/try_compile.py): ~11% cap, breaks the full generate path.
        self.rw = GdnRewinder(model)

    def _feats(self, hs):
        return torch.stack([hs[j] for j in CTX_HS], 2).flatten(2)                           # (B,L,5120)

    @torch.inference_mode()
    def prefill(self, ids):
        o = self.m(input_ids=ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        return o.past_key_values, o.logits[0, -1].float(), self._feats(o.hidden_states)

    @torch.inference_mode()
    def _forward(self, cache, tokens, pos0):
        """tokens (B, L). Returns v_lg (B,L,V), feats (B,L,5120), hs tuple, cache."""
        B, L = tokens.shape
        o = self.m(input_ids=tokens, past_key_values=cache,
                   position_ids=_pos2(pos0, L, B, self.dev),
                   use_cache=True, output_hidden_states=True, logits_to_keep=L)
        return o.logits.float(), self._feats(o.hidden_states), o.hidden_states, o.past_key_values

    def _sample(self, row):
        """One uniform -> inverse CDF over the tempered distribution (exact categorical sample).

        The drawn uniform is freshly randomised per anchor; within a draft block the uniform
        is propagated as the inverse-CDF residual (see _propose_single / nested-CDF lemma).
        """
        p = torch.softmax(row.float() / self.T, dim=-1)
        cdf = p.cumsum(-1)
        u = torch.rand((), device=self.dev, generator=self.gen)
        return torch.searchsorted(cdf, u, right=True).clamp(max=p.shape[-1] - 1)

    @torch.inference_mode()
    def _propose_multi(self, feats1, anchor, anchor_pos, n, B, u=None, mode='stratified'):
        """Draft B blocks with the drafter's inverse-CDF chain.

        `anchor` may be one token (scalar) broadcast to every block, or a (B,) tensor
        with one anchor per block (the arithmetic-coding path uses per-block anchors).
        `u` (B,) is the starting uniform of each block's chain; when None the uniforms
        are drawn here with the requested coupling (legacy callers).

        Returns Lg (B,n,V) tempered logits, ids (B,n), lps (B,), residual (B,).
        """
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        if not torch.is_tensor(anchor):
            anchor = torch.tensor(anchor, device=self.dev)
        if anchor.numel() == 1:
            anc = anchor.reshape(1, 1).expand(1, B)
        else:
            anc = anchor.view(1, B)
        o = torch.full((1, B), anchor_pos - 1, dtype=torch.long, device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anc), o)[0]                                  # (B,n,H)
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()                    # (B,n,V)
        if u is None:
            v = torch.rand(B, device=self.dev, generator=self.gen)
            if mode == 'iid':
                u = v
            elif mode == 'lattice':
                delta = torch.rand((), device=self.dev, generator=self.gen)
                u = (delta + torch.arange(B, device=self.dev, dtype=torch.float32) / B) % 1.0
            else:
                perm = torch.randperm(B, device=self.dev, generator=self.gen).float()
                u = (perm + v) / B
        else:
            u = u.to(self.dev).float().view(B)
        prev = anc.view(B)                                                                  # per-block previous token
        Lg = torch.empty(B, n, V, device=self.dev)
        ids = torch.empty(B, n, dtype=torch.long, device=self.dev)
        lps = torch.zeros(B, device=self.dev)
        for j in range(n):
            lgj = (base[:, j] + self.d.markov(prev)) / self.T                              # (B,V)
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

    @torch.inference_mode()
    def _propose_single(self, feats1, anchor, anchor_pos, n, u, mode='stratified'):
        """Single-chain draft using a GIVEN uniform u (for chain-coupled randomness).
        Returns (Lg (n,V), ids (n,), lps, u_residual). The residual is still U(0,1) by the
        nested-inverse-CDF lemma, so reusing it next round is lossless."""
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        o = torch.tensor([[anchor_pos - 1]], device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anchor.view(1, 1)), o)[0, 0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()
        prev = anchor.clone() if isinstance(anchor, torch.Tensor) else torch.tensor(anchor, device=self.dev)
        Lg = torch.empty(n, V, device=self.dev)
        ids = torch.empty(n, dtype=torch.long, device=self.dev)
        lps = torch.zeros((), device=self.dev)
        uu = float(u)
        for j in range(n):
            lgj = (base[j] + self.d.markov(prev.view(1))[0]) / self.T
            Lg[j] = lgj
            q = torch.softmax(lgj, -1)
            cdf = q.cumsum(-1)
            xj = int(torch.searchsorted(cdf, torch.tensor(uu, device=self.dev), right=True).clamp(max=V - 1))
            qx = float(q[xj].clamp_min(1e-12))
            lo = float(cdf[xj]) - qx
            lps = lps + torch.log(q[xj])
            uu = min(max((uu - lo) / qx, 0.0), 1.0 - 1e-7)
            ids[j] = xj
            prev = torch.tensor(xj, device=self.dev)
        return Lg, ids, lps, torch.tensor(uu, device=self.dev)

    def _ac_commit(self, p_lg, drafts, u):
        """Deterministic arithmetic-coding commit of one verify block.

        At slot t the target token is x*_t = F_p^{-1}(u_t) where p_t is the TARGET
        distribution at that slot. The draft token is ACCEPTED iff it equals x*_t;
        either way the committed token is x*_t, then u_t <- (u_t - F_p(x*_t))/p_t(x*_t).
        On the first mismatch the block stops (committed = x*_t); on full acceptance
        the bonus token x*_{n+1} is taken from the last row. The committed sequence is
        therefore exactly the arithmetic decode Phi_p(u) of the sequence uniform — the
        drafter only decides how many target forwards are skipped.

        p_lg (B, n+1, V) raw target logits, drafts (B, n), u (B,).
        Returns k (B,) accepted drafts, last (B,) committed final token of the round
        (correction or bonus = next anchor), u_new (B,).
        """
        B, n1, V = p_lg.shape
        n = n1 - 1
        p = torch.softmax(p_lg / self.T, -1)
        rows = torch.arange(B, device=self.dev)
        u_state = u.clone().view(B)
        k = torch.zeros(B, dtype=torch.long, device=self.dev)
        last = torch.zeros(B, dtype=torch.long, device=self.dev)
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        for t in range(n):
            cdf = p[:, t].cumsum(-1)
            xt = torch.searchsorted(cdf, u_state.unsqueeze(1), right=True).squeeze(1).clamp(max=V - 1)
            qx = p[rows, t, xt].clamp_min(1e-12)
            lo = cdf[rows, xt] - qx
            u_next = ((u_state - lo) / qx).clamp(0.0, 1.0 - 1e-7)
            active = ~done
            match = active & (drafts[:, t] == xt)
            rej = active & ~match
            k = k + match.long()
            last = torch.where(rej, xt, last)
            u_state = torch.where(active, u_next, u_state)
            done = done | rej
        cdfN = p[:, n].cumsum(-1)
        xN = torch.searchsorted(cdfN, u_state.unsqueeze(1), right=True).squeeze(1).clamp(max=V - 1)
        qxN = p[rows, n, xN].clamp_min(1e-12)
        loN = cdfN[rows, xN] - qxN
        uN = ((u_state - loN) / qxN).clamp(0.0, 1.0 - 1e-7)
        last = torch.where(done, last, xN)
        u_state = torch.where(done, u_state, uN)
        return k, last, u_state

    def _accept_multi(self, q_lg, ids, p_lg):
        """Per-slot one-uniform fused accept+correct (see dspark_fast docstring), vectorized over
        B branches and n slots; commits the DEEPEST-accepted chain. Returns (k, corr, bidx)."""
        Bn, n = ids.shape
        p = torch.softmax(p_lg / self.T, -1)
        q = torch.softmax(q_lg, -1)
        pt = p.gather(2, ids.unsqueeze(-1)).squeeze(-1)                                     # (B,n)
        qt = q.gather(2, ids.unsqueeze(-1)).squeeze(-1)
        a = (pt / qt.clamp_min(1e-12)).clamp(max=1.0)
        u = torch.rand(Bn, n, device=self.dev, generator=self.gen)
        rej = u >= a
        first = torch.cat([rej, torch.ones(Bn, 1, dtype=torch.bool, device=self.dev)], 1).int().argmax(1)
        bidx = int(first.argmax())                                                          # deepest chain
        k = int(first[bidx])
        if k == n:
            return n, None, bidx
        uk, ak = u[bidx, k], a[bidx, k]
        res = (p[bidx, k] - q[bidx, k]).clamp_min(0)
        Z = res.sum()
        if float(Z) <= 0.0:                                                                 # unreachable; guard
            return k, p[bidx, k].argmax(), bidx
        scaled = (uk - ak).clamp_min(0) / (1.0 - ak) * Z                                    # in [0, Z)
        corr = torch.searchsorted(res.cumsum(-1), scaled.unsqueeze(0))[0]
        corr = corr.clamp(max=q_lg.shape[-1] - 1)
        return k, corr, bidx

    @staticmethod
    def _expand_cache(cache, B):
        if B == 1:
            return
        for l in cache.layers:
            if _is_attn_layer(l):
                l.keys = l.keys.repeat(B, 1, 1, 1)
                l.values = l.values.repeat(B, 1, 1, 1)
            else:
                l.conv_states[0] = l.conv_states[0].repeat(B, 1, 1)
                l.recurrent_states[0] = l.recurrent_states[0].repeat(B, 1, 1, 1)

    @staticmethod
    def _select_branch(cache, k):
        """Reduce a batch-B cache back to batch 1, keeping branch k."""
        for l in cache.layers:
            if _is_attn_layer(l):
                l.keys = l.keys[k:k + 1]
                l.values = l.values[k:k + 1]
            else:
                l.conv_states[0] = l.conv_states[0][k:k + 1]
                l.recurrent_states[0] = l.recurrent_states[0][k:k + 1]

    @staticmethod
    def _branch_cache(cache, b):
        """Extract branch b of a batch-B cache into a FRESH batch-1 cache.

        Unlike deepcopy(cache)+_select_branch (which copies the whole batch-B state per
        branch — O(B^2) memory, OOM at B=64 on long sequences), this allocates only the
        batch-1 slice per layer. Tensors are cloned, not viewed, because later rounds
        mutate conv/recurrent states in place (rw.restore).
        """
        import copy as _copy
        c = _copy.copy(cache)                      # shallow cache shell
        layers = []
        for l in cache.layers:
            nl = _copy.copy(l)                     # shallow layer shell
            if _is_attn_layer(l):
                nl.keys = l.keys[b:b + 1].clone()
                nl.values = l.values[b:b + 1].clone()
            else:
                nl.conv_states = list(l.conv_states)
                nl.conv_states[0] = l.conv_states[0][b:b + 1].clone()
                nl.recurrent_states = list(l.recurrent_states)
                nl.recurrent_states[0] = l.recurrent_states[0][b:b + 1].clone()
            layers.append(nl)
        c.layers = layers
        return c

    @torch.inference_mode()
    def generate_branches(self, prompt_ids, max_new_tokens=128, n=ND, branches=4,
                          mode='stratified', return_all=True, u0=None):
        """Multi-path decode: B sequences, ONE sequence-level uniform each, Phi_p(u_b).

        Arithmetic-coding speculative decoding (the QuasiMoTTo construction): beam b's
        committed sequence is exactly Phi_p(u_b), the target's inverse-CDF (arithmetic)
        decode of its uniform. The drafter only proposes blocks; a draft token is
        accepted iff it equals the target token F_p^{-1}(u_t) induced by the running
        uniform, so drafts never change the output distribution — they only decide how
        many target forwards are skipped. Round 1 couples the B uniforms (stratified
        QMC with a random permutation, or iid); afterwards each beam evolves its own
        uniform deterministically, exactly like B independent trie traversals.

        `u0`: optional explicit uniforms (scalar or (B,)) for reproducibility/testing.
        Returns list of B (token_ids, stats) tuples (or the longest if return_all=False).
        """
        import copy as _copy

        B = max(1, int(branches))
        ids_t = torch.tensor([prompt_ids], device=self.dev)
        cache0, last0, feats0 = self.prefill(ids_t)
        pos0 = ids_t.shape[1]

        # ---- sequence-level uniforms: the QMC coupling lives entirely here ----
        if u0 is not None:
            u_seq = torch.as_tensor(u0, device=self.dev, dtype=torch.float32).reshape(-1)
            if u_seq.numel() == 1:
                u_seq = u_seq.expand(B).clone()
        else:
            v = torch.rand(B, device=self.dev, generator=self.gen)
            if mode == 'iid':
                u_seq = v
            elif mode == 'lattice':
                delta = torch.rand((), device=self.dev, generator=self.gen)
                u_seq = (delta + torch.arange(B, device=self.dev, dtype=torch.float32) / B) % 1.0
            else:
                perm = torch.randperm(B, device=self.dev, generator=self.gen).float()
                u_seq = (perm + v) / B

        # per-beam anchor = target inverse-CDF of the prefill row (consumes u)
        p0 = torch.softmax(last0.float() / self.T, -1)
        cdf0 = p0.cumsum(-1)
        anchors = torch.searchsorted(cdf0, u_seq.unsqueeze(1), right=True).squeeze(1).clamp(max=p0.numel() - 1)
        qx0 = p0[anchors].clamp_min(1e-12)
        lo0 = cdf0[anchors] - qx0
        u_seq = ((u_seq - lo0) / qx0).clamp(0.0, 1.0 - 1e-7)

        # ---- Round 1:
        Lg, dr, lps, _ = self._propose_multi(feats0[0], anchors, pos0, n, B, u=u_seq, mode=mode)

        snap = self.rw.snap(cache0)
        self._expand_cache(cache0, B)
        chunk = torch.cat([anchors.view(B, 1), dr], 1)                       # (B, n+1)
        v_lg, v_feats, hs, cache_exp = self._forward(cache0, chunk, pos0)
        k_vec, last_vec, u_vec = self._ac_commit(v_lg, dr, u_seq)

        beams = []
        for bidx in range(B):
            k = int(k_vec[bidx])
            if k == n:
                c = self._branch_cache(cache_exp, bidx)
                new_feats  = torch.cat([feats0, v_feats[bidx:bidx+1]], 1)
                new_pos    = pos0 + n + 1
                committed  = chunk[bidx]
            else:
                # FIX: start from the POST-verify cache of this branch (prefix + n+1 rows),
                # then crop back to prefix + k+1. The old code cropped the PRE-verify copy,
                # which truncated the prompt's attention KV by (n+1)-(k+1) tokens.
                c = self._branch_cache(cache_exp, bidx)
                self.rw.restore(c, snap, n_crop=(n + 1) - (k + 1))
                self.rw.rewind(c, snap, hs, k + 1, bidx=bidx)
                new_feats  = torch.cat([feats0, v_feats[bidx:bidx+1, :k+1]], 1)
                new_pos    = pos0 + k + 1
                committed  = chunk[bidx, :k+1]
            beams.append({
                'cache': c, 'feats': new_feats, 'anchor': last_vec[bidx],
                'pos': new_pos, 'out': [committed], 'u': u_vec[bidx],
                'st': {'rounds': 1, 'draft_ms': 0.0, 'verify_ms': 0.0, 'rewind_ms': 0.0,
                       'forwards': 1, 'accepted': [k], 'tokens': int(committed.numel()),
                       'all_accepted': int(k == n), 'draft_lp': float(lps[bidx])}
            })

        del cache_exp
        torch.cuda.empty_cache()

        # ---- Round 2+: single-chain arithmetic-coding decode per beam ----
        for bm in beams:
            cache = bm['cache']
            feats = bm['feats']
            anchor = bm['anchor']
            pos = bm['pos']
            u_b = bm['u']
            out = bm['out']
            st = bm['st']
            t_rew = 0.0

            while sum(t.numel() for t in out) < max_new_tokens:
                if bool((anchor == self.tok.eos_token_id).any()):
                    out.append(anchor.view(1))
                    break

                t0 = time.perf_counter()
                Lg1, dr1, lp1, _ = self._propose_single(feats[0], anchor, pos, n, u_b, mode=mode)
                torch.cuda.synchronize()
                st['draft_ms'] += (time.perf_counter() - t0) * 1e3

                snap1 = self.rw.snap(cache)
                t0 = time.perf_counter()
                chunk1 = torch.cat([anchor.view(1), dr1]).view(1, -1)
                v_lg1, v_feats1, hs1, cache = self._forward(cache, chunk1, pos)
                k1_t, last1_t, u1_t = self._ac_commit(v_lg1, dr1.unsqueeze(0), u_b.view(1))
                k1 = int(k1_t[0])
                anchor_next = last1_t[0]
                u_b = u1_t[0]
                torch.cuda.synchronize()
                st['verify_ms'] += (time.perf_counter() - t0) * 1e3
                st['forwards'] += 1
                st['rounds']   += 1
                st['accepted'].append(k1)

                if k1 == n:
                    st['all_accepted'] += 1
                    feats = torch.cat([feats, v_feats1[0:1]], 1)
                    pos += n + 1
                    committed1 = chunk1[0]
                else:
                    committed1 = chunk1[0, :k1+1]
                    t0 = time.perf_counter()
                    self.rw.restore(cache, snap1, n_crop=(n+1)-(k1+1))
                    self.rw.rewind(cache, snap1, hs1, k1+1, bidx=0)
                    torch.cuda.synchronize()
                    t_rew += time.perf_counter() - t0
                    feats = torch.cat([feats, v_feats1[0:1, :k1+1]], 1)
                    pos += k1 + 1
                anchor = anchor_next

                out.append(committed1)
                st['tokens'] = sum(t.numel() for t in out)
                if bool((committed1 == self.tok.eos_token_id).any()):
                    break
                if bool((anchor == self.tok.eos_token_id).any()):
                    out.append(anchor.view(1))
                    break
            else:
                if sum(t.numel() for t in out) < max_new_tokens:
                    out.append(anchor.view(1))

            st['rewind_ms'] = t_rew * 1e3
            bm['final'] = torch.cat(out)[:max_new_tokens].tolist()
            st['tokens'] = len(bm['final'])

        results = [(bm['final'], bm['st']) for bm in beams]
        if return_all:
            return results
        best = max(results, key=lambda x: x[1]['tokens'])
        return best[0], best[1]


    def generate_tree(self, prompt_ids, max_new_tokens=128, n=ND, branches=4, mode='stratified',
                       max_beams=8, prune='median'):
        """LEGACY tree-parallel decoding with beam pruning (standard speculative commit).

        NOTE: this path predates the arithmetic-coding rewrite. It still uses the classical
        min(1,p/q) accept/correct with fresh randomness, so its output is NOT Phi_p(u) and
        its chains are not coupled by a sequence uniform. It contains no Gumbel noise, but
        for paper-faithful sampling use generate() / generate_branches() instead.

        Each active beam drafts B chains → batch-verifies → forks into B child beams.
        Pruning strategies (prune=):
          'median' : drop k < median(k)  — prunes ~half, stable beam count
          'mean'   : drop k < mean(k)
          'nonzero': drop k == 0 only    — gentler, keeps all beams that accepted anything
        Survivors capped at max_beams; at most max_beams target forwards per step.

        Cache management: on rejection, child reuses the parent's snap + GDN rewind
        (no full deepcopy); on full-accept, child is a slice of the already-computed
        expanded cache (also no extra deepcopy). One deepcopy per PARENT beam per step
        (for the snap), not per child.

        GDN states per beam are independent (batch-dim isolation, no tree attention).
        """
        import copy as _copy
        import statistics as _stat

        B = max(1, int(branches))
        ids_t = torch.tensor([prompt_ids], device=self.dev)
        cache0, last0, feats0 = self.prefill(ids_t)
        pos0 = ids_t.shape[1]
        beams = [{'cache': cache0, 'feats': feats0, 'anchor': self._sample(last0),
                  'pos': pos0, 'out': []}]
        st = {'rounds': 0, 'draft_ms': 0.0, 'verify_ms': 0.0, 'rewind_ms': 0.0,
              'forwards': 0, 'accepted': [], 'tokens': 0, 'all_accepted': 0,
              'pruned': 0, 'beams_per_round': []}
        t_rew = 0.0

        def total_toks(bl):
            return sum(sum(t.numel() for t in b['out']) for b in bl)

        while beams and total_toks(beams) < max_new_tokens:
            children = []

            for bm in beams:
                anchor, pos, feats, cache = bm['anchor'], bm['pos'], bm['feats'], bm['cache']

                t0 = time.perf_counter()
                Lg, dr, lps, _ = self._propose_multi(feats[0], anchor, pos, n, B, mode=mode)
                torch.cuda.synchronize()
                st['draft_ms'] += (time.perf_counter() - t0) * 1e3

                # ONE deepcopy per parent (for the GDN snap shared across all children)
                snap = self.rw.snap(cache)
                cache_exp = _copy.deepcopy(cache)
                self._expand_cache(cache_exp, B)
                t0 = time.perf_counter()
                chunk = torch.cat([anchor.view(1, 1).expand(B, 1), dr], 1)
                v_lg, v_feats, hs, cache_exp = self._forward(cache_exp, chunk, pos)
                torch.cuda.synchronize()
                st['verify_ms'] += (time.perf_counter() - t0) * 1e3
                st['forwards'] += 1
                st['rounds']   += 1

                p_lg = v_lg[:, :n]
                p = torch.softmax(p_lg / self.T, -1)
                q = torch.softmax(Lg, -1)
                pt = p.gather(2, dr.unsqueeze(-1)).squeeze(-1)
                qt = q.gather(2, dr.unsqueeze(-1)).squeeze(-1)
                a_acc = (pt / qt.clamp_min(1e-12)).clamp(max=1.0)
                u = torch.rand(B, n, device=self.dev, generator=self.gen)
                rej = u >= a_acc
                first = torch.cat([rej, torch.ones(B, 1, dtype=torch.bool, device=self.dev)],
                                   1).int().argmax(1)

                for bidx in range(B):
                    k = int(first[bidx])
                    if k == n:
                        # full accept: extract this branch's batch-1 slice (no O(B^2) copy)
                        c_child = self._branch_cache(cache_exp, bidx)
                        new_feats  = torch.cat([feats, v_feats[bidx:bidx + 1]], 1)
                        new_pos    = pos + n + 1
                        committed  = chunk[bidx]
                        new_anchor = self._sample(v_lg[bidx, n])
                        st['all_accepted'] += 1
                    else:
                        # partial: restore from snap + GDN rewind — no extra deepcopy of snap
                        c_child = _copy.deepcopy(cache)   # fresh copy of pre-verify state
                        uk, ak = u[bidx, k], a_acc[bidx, k]
                        res = (p[bidx, k] - q[bidx, k]).clamp_min(0)
                        Z = res.sum()
                        if float(Z) <= 0.0:
                            corr = p[bidx, k].argmax()
                        else:
                            scaled = (uk - ak).clamp_min(0) / (1.0 - ak) * Z
                            corr = torch.searchsorted(res.cumsum(-1), scaled.unsqueeze(0))[0]
                            corr = corr.clamp(max=v_lg.shape[-1] - 1)
                        t0 = time.perf_counter()
                        self.rw.restore(c_child, snap, n_crop=(n + 1) - (k + 1))
                        self.rw.rewind(c_child, snap, hs, k + 1, bidx=bidx)
                        torch.cuda.synchronize()
                        t_rew += time.perf_counter() - t0
                        new_feats  = torch.cat([feats, v_feats[bidx:bidx + 1, :k + 1]], 1)
                        new_pos    = pos + k + 1
                        committed  = chunk[bidx, :k + 1]
                        new_anchor = corr

                    st['accepted'].append(k)
                    eos_hit = (bool((committed == self.tok.eos_token_id).any()) or
                               bool((new_anchor == self.tok.eos_token_id).any()))
                    children.append({'cache': c_child, 'feats': new_feats,
                                      'anchor': new_anchor, 'pos': new_pos,
                                      'out': bm['out'] + [committed], 'k': k, 'eos': eos_hit})

            st['beams_per_round'].append(len(children))

            # -- prune --
            if len(children) > 1:
                ks = [c['k'] for c in children]
                if prune == 'median':
                    thr = _stat.median(ks)
                elif prune == 'mean':
                    thr = sum(ks) / len(ks)
                else:  # nonzero: only drop k==0
                    thr = 0.5
                kept = [c for c in children if c['k'] >= thr]
                if not kept:
                    kept = [max(children, key=lambda c: c['k'])]
                st['pruned'] += len(children) - len(kept)
                if len(kept) > max_beams:
                    kept.sort(key=lambda c: -c['k'])
                    kept = kept[:max_beams]
                children = kept

            live = [c for c in children if not c['eos']]
            beams = live if live else []
            if not beams:
                break

        st['rewind_ms'] = t_rew * 1e3
        pool = beams if beams else children
        best = max(pool, key=lambda b: sum(t.numel() for t in b['out']))
        all_toks = torch.cat(best['out'])[:max_new_tokens].tolist()
        st['tokens'] = len(all_toks)
        return all_toks, st


    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=128, n=ND, branches=1, mode='stratified', u0=None):
        """Single-chain arithmetic-coding decode: the output is exactly Phi_p(u0).

        Same commit rule as generate_branches: at every position the committed token is
        the target's inverse-CDF token for the running uniform; the drafter only lets us
        skip target forwards when its proposal matches.
        """
        ids = torch.tensor([prompt_ids], device=self.dev)
        cache, last, feats = self.prefill(ids)
        pos = ids.shape[1]
        st = {'rounds': 0, 'draft_ms': 0.0, 'verify_ms': 0.0, 'rewind_ms': 0.0, 'forwards': 0,
              'accepted': [], 'tokens': 0, 'all_accepted': 0, 'draft_lp': 0.0}
        if u0 is None:
            u = torch.rand((), device=self.dev, generator=self.gen)
        else:
            u = torch.as_tensor(u0, device=self.dev, dtype=torch.float32).reshape(())

        # first anchor = target inverse-CDF of the prefill row
        p0 = torch.softmax(last.float() / self.T, -1)
        cdf0 = p0.cumsum(-1)
        anchor = torch.searchsorted(cdf0, u.reshape(1), right=True)[0].clamp(max=p0.numel() - 1)
        qx0 = p0[anchor].clamp_min(1e-12)
        lo0 = cdf0[anchor] - qx0
        u = ((u - lo0) / qx0).clamp(0.0, 1.0 - 1e-7)

        out = []
        t_rew = 0.0
        while sum(t.numel() for t in out) < max_new_tokens:
            if bool((anchor == self.tok.eos_token_id).any()):
                out.append(anchor.view(1))
                break
            t0 = time.perf_counter()
            Lg, dr, lps, _ = self._propose_single(feats[0], anchor, pos, n, u, mode=mode)
            torch.cuda.synchronize()
            st['draft_ms'] += (time.perf_counter() - t0) * 1e3

            snap = self.rw.snap(cache)
            t0 = time.perf_counter()
            chunk = torch.cat([anchor.view(1), dr]).view(1, -1)
            v_lg, v_feats, hs, cache = self._forward(cache, chunk, pos)
            k_t, last_t, u_t = self._ac_commit(v_lg, dr.unsqueeze(0), u.view(1))
            k = int(k_t[0]); anchor_next = last_t[0]; u = u_t[0]
            torch.cuda.synchronize()
            st['verify_ms'] += (time.perf_counter() - t0) * 1e3
            st['forwards'] += 1
            st['rounds'] += 1
            st['accepted'].append(k)
            st['draft_lp'] = float(lps)

            if k == n:
                st['all_accepted'] += 1
                feats = torch.cat([feats, v_feats[0:1]], 1)
                pos += n + 1
                committed = chunk[0]
            else:
                committed = chunk[0, :k+1]
                t0 = time.perf_counter()
                self.rw.restore(cache, snap, n_crop=(n+1)-(k+1))
                self.rw.rewind(cache, snap, hs, k+1, bidx=0)
                torch.cuda.synchronize()
                t_rew += time.perf_counter() - t0
                feats = torch.cat([feats, v_feats[0:1, :k+1]], 1)
                pos += k+1
            anchor = anchor_next
            out.append(committed)
            st['tokens'] = sum(t.numel() for t in out)
            if bool((committed == self.tok.eos_token_id).any()):
                break
            if bool((anchor == self.tok.eos_token_id).any()):
                out.append(anchor.view(1))
                break
        st['rewind_ms'] = t_rew * 1e3
        return torch.cat(out)[:max_new_tokens].tolist(), st
