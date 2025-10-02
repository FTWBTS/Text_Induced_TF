# TICALPlugin.py
import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from layers.TICAL import Text2Kernels, KernelEmbed, FutureDeltaProj, causal_cost, sinkhorn, delta_contrastive

import os, torch, random, numpy as np

def set_all_seeds(seed: int = 0, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # CUDA 11+：为了一些 cuBLAS 算法的确定性
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或 ":4096:8"

        # 可选：限制 matmul 精度，降低硬件相关差异
        try:
            torch.set_float32_matmul_precision("high")  # 或 "medium"
        except Exception:
            pass

class TICALPlugin(nn.Module):
    def __init__(self, configs,freeze_future_proj: bool = False):
        set_all_seeds(0,True)
        super().__init__()

        self.txt_dim = configs.text_emb
        self.K = configs.TICAL_k
        self.shape_types = configs.TICAL_shape_types
        self.kernel_emb_dim = configs.TICAL_kernel_emb_dim
        self.cot_eps = configs.TICAL_cot_eps
        self.cot_iters = configs.TICAL_cot_iters
        self.cot_bandwidth = configs.TICAL_cot_bandwidth
        self.cot_alpha = configs.TICAL_cot_alpha
        self.cot_beta = configs.TICAL_cot_beta
        self.gating_dim = configs.TICAL_gate_dim
        self.gate_weight = configs.TICAL_gate_weight
        self.kernel_H_scale = configs.TICAL_kernel_H_scale
        self.lmb_mse = 1.0
        self.lmb_cot = configs.TICAL_lmb_cot
        self.lmb_delta = configs.TICAL_lmb_delta
        self.lmb_entropy = configs.TICAL_lmb_entropy
        self.lmb_tv = configs.TICAL_lmb_tv
    

        # Text to Temporal Kernels
        self.text2k = Text2Kernels(self.txt_dim, K=self.K, shape_types=self.shape_types, H_scale=self.kernel_H_scale)
        # Kernel params -> embedding (for COT/contrast)
        self.kemb = KernelEmbed(K=self.K, H = configs.pred_len,dim=self.kernel_emb_dim)
        # TIR 1->D expansion
        self.tir_expand = nn.Linear(1, configs.enc_in, bias=True)
        nn.init.xavier_uniform_(self.tir_expand.weight)
        nn.init.zeros_(self.tir_expand.bias)

        # Per-variable hint (kept same as PatchTST version)
        self.tir_var_proj = nn.Parameter(torch.ones(configs.enc_in))

        # Per-variable gating
        
        gate_in = configs.enc_in + self.txt_dim + configs.enc_in
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, 512), nn.ReLU(),
            nn.Linear(512, configs.enc_in)
        )

        # Future delta projection for auxiliary objectives
        self.future_proj = FutureDeltaProj(in_dim=3 * configs.enc_in, out_dim=self.kernel_emb_dim)
        
        for p in self.future_proj.parameters():
            p.requires_grad = not freeze_future_proj

    def _text_induced_response(self, text_emb: torch.Tensor, H: int):
        kappa, params = self.text2k(text_emb, H)  # [B, K, H]  kappa是核形状混合
        if kappa is None:
            return None, None, None
        y_tir_scalar = kappa.sum(dim=1)  # [B, H]  y_tir_scalar是事件响应曲线
        return y_tir_scalar, kappa, params

    def _fuse(self, y_res: torch.Tensor, y_tir_scalar: torch.Tensor,
              enc_out: torch.Tensor, text_emb: torch.Tensor):
        """
        Fusion of residual prediction with text-induced response
        """
        if y_tir_scalar is None:
            return y_res, None

        B, H, D = y_res.shape
        # print(D)
        # Expand TIR to D vars
        y_tir = self.tir_expand(y_tir_scalar.unsqueeze(-1))  # [B,H,1] -> [B,H,D]
        
        # print("y_tir.shape:",y_tir.shape)
        y_tir = y_tir * self.tir_var_proj[None, None, :]
        
        # print("y_tir after shape:",y_tir.shape)

        # text mean
        if text_emb.dim() == 3:
            e_mean = text_emb.mean(dim=1)  # [B, txt_dim]
        else:
            e_mean = text_emb  # [B, txt_dim]

        # ts global from encoder (mean over variables)
        ts_global = enc_out.mean(dim=1)  # [B, d_model]
        

        var_hint = self.tir_var_proj[None, ...]  # [1,D]
        # print("shape shape:",ts_global.shape, e_mean.shape, self.tir_var_proj.shape)
        
        ##全局时序信息，全局文本信息，事件响应曲线 进行拼接
        gate_in_all = torch.cat([ts_global, e_mean, var_hint.expand(B, -1)], dim=-1)  

        gate = torch.sigmoid(self.gate_mlp(gate_in_all))  # [B, D]
        gate = gate.view(B, 1, D)
        
        # print("gate:",gate.shape)

        y_hat = y_res + self.gate_weight * gate * (y_tir - y_res)
        return y_hat, gate

    def _aux_losses(self, kappa: torch.Tensor, params: Dict[str, torch.Tensor],
                    y_future: torch.Tensor, H: int):
        if kappa is None or y_future is None:
            return {}

        # print("params",params)
        k_emb = self.kemb(kappa)  # [B, K, Dk]，将不同文本核的不同形态映射到一个嵌入空间,一个事件的影响是由多个形态叠加形成的文本核
        with torch.set_grad_enabled(self.future_proj.proj.weight.requires_grad):
            f_pos = self.future_proj(y_future)  # [B, H, Dk]

        C = causal_cost(k_emb, f_pos, params['mu'], alpha=self.cot_alpha, beta=self.cot_beta)  # [B, K, H]

        with torch.no_grad():
            r = (kappa.sum(-1) + 1e-6)  # [B, K]
            r = r / r.sum(dim=-1, keepdim=True) ###实现非均匀边际
            c = torch.full((C.size(0), C.size(2)), 1.0 / C.size(2), device=C.device, dtype=C.dtype)#为每个样本生成一个 均匀的列边际，即每个未来时刻的质量（或解释份额）都相等

        Pi = sinkhorn(C, eps=self.cot_eps, iters=self.cot_iters, r=r, c=c)  # [B, K, H]

        # Negatives
        shift = max(1, H // 6)
        f_neg1 = torch.roll(f_pos, shifts=shift, dims=1)
        # print("f_nag1.shape",f_neg1.shape)
        perm = torch.randperm(f_pos.size(-1), device=f_pos.device)
        f_neg2 = f_pos[..., perm]
        # print("f_nag2.shape",f_neg2.shape)

        loss_delta = delta_contrastive(k_emb, f_pos, Pi, negatives=[f_neg1, f_neg2], tau=0.07)
        loss_cot = (C * Pi).sum(dim=(1, 2)).mean()

        shape_w = params['shape_w']  # [B, K, S]
        ent = -(shape_w.clamp_min(1e-8) * shape_w.clamp_min(1e-8).log()).sum(dim=-1).mean()
        #用于控制核的形状混合权重（shape_w）的分布，使得模型不倾向于将所有质量集中到某一个形状（例如，所有核都变成高斯或步进形状），而是促进“多样性”或“平衡”的形状组合。

        mu = params['mu']  # [B, K]
        tv = (mu[:, 1:] - mu[:, :-1]).abs().mean()
        #避免滞后期 μ 的剧烈变化，使得相邻的核滞后期不至于相差过大，从而使得模型的行为更加稳定、可解释。


        return {
            'loss_cot': loss_cot,
            'loss_delta': loss_delta,
            'loss_entropy': ent,
            'loss_tv': tv,
            'Pi': Pi.detach()
        }
