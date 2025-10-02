from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F



# ======================= TICaL Core Blocks =======================

class Text2Kernels(nn.Module):
    """
    e_txt -> K kernels over the prediction horizon H with a small shape bank.
    Same as your PatchTST version (minor refactor for reuse).
    """
    def __init__(self, txt_dim: int, K: int = 4, shape_types: int = 4, H_scale: float = 0.2):
        super().__init__()
        self.txt_dim = txt_dim
        self.K = K
        self.S = shape_types  # gaussian / expdecay / step / bipeak
        hid = max(256, txt_dim)
        self.H_scale = H_scale
        self.mlp = nn.Sequential(
            nn.Linear(txt_dim, hid), nn.ReLU(),
            nn.Linear(hid, 512), nn.ReLU(),
            nn.Linear(512, K * (4 + self.S))  # mu, sigma, a, tau + shape_logits[S]
        )

    @staticmethod
    def _build_bank(l, mu, sigma, tau):
        gauss  = torch.exp(-0.5 * ((l - mu) / (sigma + 1e-9)).pow(2))
        expdc  = torch.exp(-(torch.relu(l - mu)) / (tau + 1e-9))
        step   = (l >= mu).float()
        bipeak = torch.exp(-0.5 * ((l - (mu - sigma)) / (sigma + 1e-9)).pow(2)) + \
                torch.exp(-0.5 * ((l - (mu + sigma)) / (sigma + 1e-9)).pow(2))
        lininc = torch.relu((l - mu) / (tau + 1e-9))   # 线性增加
        expinc = 1 - torch.exp(-(torch.relu(l - mu)) / (tau + 1e-9))  # 指数增加
        sinus  = torch.sin(2 * math.pi * (l - mu) / (tau + 1e-9))     # 周期波动
        shock  = torch.exp(-(torch.relu(l - mu)) / (tau + 1e-9))      # 冲击后衰退
        
        return torch.stack([gauss, expdc, step, bipeak, lininc, expinc, sinus, shock], dim=-1)


    def forward(self, e_txt: torch.Tensor, H: int):
        """
        e_txt: [B, txt_dim] or [B, Q, txt_dim]
        return:
          kappa: [B, K, H]
          params: dict(mu,sigma,a,tau,shape_w) with shapes [B,K] and [B,K,S]
        """
        if e_txt is None:
            return None, None

        if e_txt.dim() == 3:
            B, Q, D = e_txt.shape #【Batch Size,同一样本的Q条描述，每条描述的embeding维度】
            e = e_txt.reshape(B*Q, D)
            multi_text = True
        elif e_txt.dim() == 2:
            B, D = e_txt.shape
            e = e_txt
            Q = 1
            multi_text = False
        else:
            raise ValueError("text_emb must be [B,txt_dim] or [B,Q,txt_dim]")

        raw = self.mlp(e)  # [B*Q, K*(4+S)]
        mu, sigma, a, tau, shape_logits = torch.split(
            raw, [self.K, self.K, self.K, self.K, self.K*self.S], dim=-1
        )
        #五个参数分别代表的意思：μ (中心位置 = 期望延迟时间)，σ (扩散程度 = 波动范围)，a (幅度)，τ (指数衰减速率)，shape_logits (混合不同形状核的权重)
        #做物理域的映射
        mu_n    = torch.sigmoid(mu)  #将中心位置归一到0-1
        sigma_n = F.softplus(sigma) + 1e-3  #保证值是正数
        tau_n   = F.softplus(tau) + 1e-3    #保证值是正数
        a_pos   = F.softplus(a)             #保证值是正数
        shape_w = F.softmax(shape_logits.view(-1, self.K, self.S), dim=-1)  #K个核，每个核的S种混合权重之和为1

        #将长度扩充到预测长度：H
        l = torch.arange(1, H+1, device=e.device, dtype=e.dtype)[None, None, :]  # [1,1,H]
        muH     = mu_n * H
        sigmaH  = sigma_n * (self.H_scale * H)
        tauH    = tau_n   * (self.H_scale * H)

        bank = self._build_bank(l, muH[..., None], sigmaH[..., None], tauH[..., None])  # [B*Q,K,H,S]
        kappa = (bank * shape_w[..., None, :]).sum(-1)  # [B*Q,K,H]  # 核形状混合
        kappa = kappa / (kappa.sum(dim=-1, keepdim=True) + 1e-6)
        kappa = a_pos[..., None] * kappa  # amplitude
        # print("kappa shape:",kappa.shape)
        if multi_text:
            kappa  = kappa.view(B, Q, self.K, H).sum(dim=1) / math.sqrt(Q)
            muH    = muH.view(B, Q, self.K).mean(dim=1)
            sigmaH = sigmaH.view(B, Q, self.K).mean(dim=1)
            a_pos  = a_pos.view(B, Q, self.K).mean(dim=1)
            tauH   = tauH.view(B, Q, self.K).mean(dim=1)
            shape_w= shape_w.view(B, Q, self.K, self.S).mean(dim=1)
        else:
            kappa  = kappa.view(B, self.K, H)
            muH    = muH.view(B, self.K)
            sigmaH = sigmaH.view(B, self.K)
            a_pos  = a_pos.view(B, self.K)
            tauH   = tauH.view(B, self.K)
            shape_w= shape_w.view(B, self.K, self.S)

        params = dict(mu=muH, sigma=sigmaH, a=a_pos, tau=tauH, shape_w=shape_w)
        return kappa, params


# class KernelEmbed(nn.Module):
#     """ Map kernel params to embeddings for COT/contrast. """
#     def __init__(self, K: int, S: int, dim: int = 64):
#         super().__init__()
#         inp = 4 + S  # mu,sigma,a,tau + shape_mix(S)
#         self.fc = nn.Sequential(
#             nn.Linear(inp, 128), nn.ReLU(),
#             nn.Linear(128, dim)
#         )
#         self.K = K
#         self.S = S
#         self.dim = dim

#     def forward(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
#         mu     = params['mu']      # [B,K]
#         sigma  = params['sigma']
#         a      = params['a']
#         tau    = params['tau']
#         shape_w= params['shape_w']  # [B,K,S]
#         H_approx = mu.size(-1) if isinstance(mu.size(-1), int) else 1
#         mu_n   = mu / (mu.new_tensor(H_approx) + 1e-9)
#         sigma_n= torch.log1p(sigma)
#         tau_n  = torch.log1p(tau)
#         feat = torch.cat([
#             mu_n.unsqueeze(-1),
#             sigma_n.unsqueeze(-1),
#             F.softplus(a).unsqueeze(-1),
#             tau_n.unsqueeze(-1),
#             shape_w
#         ], dim=-1)  # [B,K,4+S]
#         emb = self.fc(feat)    # [B,K,Dk]
#         return emb


class KernelEmbed(nn.Module):
    """ Map kernel params to embeddings for COT/contrast using kappa. """
    def __init__(self, K: int, H: int, dim: int = 64):
        super().__init__()
        # K: number of kernels, H: time length
        # Now we are directly working with kappa, so we will expect kappa shape [B, K, H]
        # The input size now is H for each kernel, plus 4 additional parameters (mu, sigma, a, tau)
        self.fc = nn.Sequential(
            nn.Linear(H, 128),  # H (from kappa) + 4 parameters (mu, sigma, a, tau)
            nn.ReLU(),
            nn.Linear(128, dim)  # Output embedding of size `dim`
        )
        self.K = K
        self.H = H
        self.dim = dim

    def forward(self,kappa: torch.Tensor) -> torch.Tensor:
        """
        kappa: [B, K, H], the kernel response for each time step

        Returns:
          emb: [B, K, Dk], embedding for each kernel
        """
        # print("kappa shape:",kappa.shape)
        emb = self.fc(kappa)    # [B, K, Dk]
        
        return emb


# class FutureDeltaProj(nn.Module):
#     """ Trainable projection for future delta features. """
#     def __init__(self, in_dim: int, out_dim: int):
#         super().__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         nn.init.xavier_uniform_(self.proj.weight)
#         nn.init.zeros_(self.proj.bias)

#     def forward(self, y_future: torch.Tensor) -> torch.Tensor:
#         # y_future: [B,H,D]
#         dy  = y_future[:, 1:, :] - y_future[:, :-1, :]
#         ddy = dy[:, 1:, :] - dy[:, :-1, :]
#         B, H, D = y_future.shape
#         pad_dy  = F.pad(dy,  (0,0,1,0))   # [B,H,D]
#         pad_ddy = F.pad(ddy, (0,0,2,0))   # [B,H,D]
#         feats = torch.cat([pad_dy, pad_ddy, y_future], dim=-1)  # [B,H,3D]
#         return self.proj(feats)  # [B,H,out_dim]


class FutureDeltaProj(nn.Module):
    """ Trainable projection for future delta features with CNN fusion. """
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        # feats 拼接后维度就是 3*D
        self.in_dim = in_dim
        self.out_dim = out_dim
        padding = (kernel_size - 1) // 2
        self.cnn = nn.Conv1d(self.in_dim, self.in_dim,kernel_size=kernel_size, padding=padding)

        # 投影层
        self.proj = nn.Linear(self.in_dim, out_dim)

        # 初始化
        nn.init.kaiming_uniform_(self.cnn.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, y_future: torch.Tensor) -> torch.Tensor:
        # [B,H,D]
        dy  = y_future[:, 1:, :] - y_future[:, :-1, :]
        ddy = dy[:, 1:, :] - dy[:, :-1, :]
        pad_dy  = F.pad(dy,  (0,0,1,0))
        pad_ddy = F.pad(ddy, (0,0,2,0))

        feats = torch.cat([pad_dy, pad_ddy, y_future], dim=-1)  # [B,H,3D]

        # CNN 融合
        x = feats.transpose(1, 2)        # [B,3D,H]
        x = F.relu(self.cnn(x))          # [B,3D,H]
        x = x.transpose(1, 2)            # [B,H,3D]
        return self.proj(x)              # [B,H,out_dim]



def causal_cost(kernel_emb: torch.Tensor,
                future_emb: torch.Tensor,
                mu: torch.Tensor,
                alpha: float = 1.0,
                beta: float = 0.02) -> torch.Tensor:
    """
    kernel_emb: [B,K,Dk]
    future_emb: [B,H,Dk]
    mu: [B,K]  expected lag position (1..H)
    return C: [B,K,H]
    """
    k = F.normalize(kernel_emb, dim=-1)
    f = F.normalize(future_emb, dim=-1)
    cos = 1 - torch.einsum('bkd,bhd->bkh', k, f)  # [B,K,H] 余弦距离，核K和未来H的相似度
    idx = torch.arange(future_emb.size(1), device=mu.device).float()[None, None, :] + 1.0
    penalty = torch.relu(mu[..., None] - idx)  # mu[...,none]是代表每个核的期望中心位置，idx是未来时间步的位置,relu用来截断负值（只保留idx<mu）的情况，这样对于过早的匹配做惩罚
    return alpha * cos + beta * penalty


def sinkhorn(C: torch.Tensor,
             eps: float = 0.1,
             iters: int = 50,
             band_mask: Optional[torch.Tensor] = None,
             r: Optional[torch.Tensor] = None,
             c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Entropic OT Sinkhorn with optional band mask and non-uniform marginals.
    C: [B,K,H], band_mask: [B,K,H]
    r: [B,K] row marginals, c: [B,H] column marginals
    return Pi: [B,K,H]
    """
    B, K, H = C.shape
    if r is None:
        r = torch.full((B, K), 1.0 / K, device=C.device, dtype=C.dtype)
    if c is None:
        c = torch.full((B, H), 1.0 / H, device=C.device, dtype=C.dtype)

    #C是对齐成本矩阵，两部分：第一部分是核和未来时间步的余弦距离（表示相似度），第二部分是对过早匹配的惩罚
    #Kmat 是从C来的加权传输代价矩阵，表示源和目标的关系强度
    # print("C shape",C.shape)
    Kmat = torch.exp(-C / eps)  # Gibbs
    if band_mask is not None:
        Kmat = Kmat * band_mask

    u = torch.ones_like(r)
    v = torch.ones_like(c)
    
    #转成软对齐概率（Sinkhorn 最优传输
    for _ in range(iters):
        Kv = torch.bmm(Kmat, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)   # [B,K]
        u = r / Kv
        Ku = torch.bmm(Kmat.transpose(1,2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)  # [B,H]
        v = c / Ku
    Pi = u.unsqueeze(-1) * Kmat * v.unsqueeze(1)
    #Pi[b,k,t] 就是第 k 个核对齐到时间步 t 的“软匹配权重”。
    return Pi


def delta_contrastive(kernel_emb: torch.Tensor,
                      future_emb_pos: torch.Tensor,
                      Pis: torch.Tensor,
                      negatives=None,
                      tau: float = 0.07) -> torch.Tensor:
    """
    kernel_emb: [B,K,Dk]
    future_emb_pos: [B,H,Dk]
    Pis: [B,K,H] OT plan
    negatives: list of [B,H,Dk]
    return CE loss (scalar)
    """
    if negatives is None:
        negatives = []

    ###引入Pi作为权重，对相似度进行加权求和,让“应该对齐/合理对齐”的位置贡献更大，强化“合理匹配”
    def agg_sim(k, f, Pi):
        sim_mat = torch.einsum('bkd,bhd->bkh', F.normalize(k, dim=-1), F.normalize(f, dim=-1))
        return (Pi * sim_mat).sum(dim=(1,2))  # [B]

    pos = agg_sim(kernel_emb, future_emb_pos, Pis)  # [B]
    logits = [pos / tau]
    for fneg in negatives:
        neg = agg_sim(kernel_emb, fneg, Pis)
        logits.append(neg / tau)
    logits = torch.stack(logits, dim=-1)  # [B, 1+Nneg]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
