import torch
import torch.nn.functional as F
from dspark_fast import DSparkDecodeFast

class S5VariantEngine(DSparkDecodeFast):
    def propose_variant(self, feats1, anchor, anchor_pos, n, u, mode='update'):
        V = self.d.lm_head.weight.shape[0]
        hctx = self.d.encode_ctx(feats1[:anchor_pos].unsqueeze(0))
        o = torch.tensor([[anchor_pos - 1]], device=self.dev)
        z = self.d.backbone(hctx, self.d.embed(anchor.view(1, 1)), o)[0, 0]
        base = self.d.lm_head(z.to(self.d.lm_head.weight.dtype)).float()
        prev = anchor.clone() if isinstance(anchor, torch.Tensor) else torch.tensor(anchor, device=self.dev)
        Lg = torch.empty(n, V, device=self.dev)
        ids = torch.empty(n, dtype=torch.long, device=self.dev)
        
        uu = float(u)
        
        for j in range(n):
            lgj = (base[j] + self.d.markov(prev.view(1))[0]) / self.T
            Lg[j] = lgj
            
            if mode == 'gumbel':
                # Gumbel-Max: no u involved
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(lgj) + 1e-10) + 1e-10)
                xj = (lgj + gumbel_noise).argmax()
            else:
                # Inverse-CDF
                q = torch.softmax(lgj, -1)
                cdf = q.cumsum(-1)
                xj = int(torch.searchsorted(cdf, torch.tensor(uu, device=self.dev), right=True).clamp(max=V - 1))
                
                if mode == 'update':
                    qx = float(q[xj].clamp_min(1e-12))
                    lo = float(cdf[xj]) - qx
                    uu = min(max((uu - lo) / qx, 0.0), 1.0 - 1e-7)
            
            ids[j] = xj
            prev = torch.tensor(xj, device=self.dev)
            
        return Lg, ids, torch.zeros((), device=self.dev), u # Always return original u
