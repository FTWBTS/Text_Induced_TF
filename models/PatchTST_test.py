from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# You must have these in your environment as in your original codebase:
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.TICAL import Text2Kernels, KernelEmbed, FutureDeltaProj, causal_cost, sinkhorn, delta_contrastive


# ====================== PatchTST Backbone + Improved TICaL ======================

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)       # [bs x nvars x nf]
        x = self.linear(x)        # [bs x nvars x target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST + Improved TICaL
    - Residual forecasting branch (PatchTST head)
    - Text -> Temporal kernels -> TIR (per-kernel normalized, positive amplitude)
    - TIR expanded to multivariate via Linear(1->D)
    - Per-variable gating (B×1×D), convex mixing in output space
    - Optional COT + Delta-Contrastive auxiliary losses
    """
    def __init__(self, configs, patch_len=16, stride=8,
                 use_text=True, txt_dim=768, K=4, shape_types=4,
                 kernel_emb_dim=32, cot_eps=0.1, cot_iters=50, cot_bandwidth=6,
                 cot_alpha=1.0, cot_beta=0.02,
                 lmb_cot=1e-3, lmb_delta=1e-3, lmb_entropy=1e-6, lmb_tv=1e-6,
                 kernel_H_scale=0.2,
                 freeze_future_proj: bool = False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # number of variables D
        self.use_text = use_text
        self.txt_dim = txt_dim
        self.K = K
        self.shape_types = shape_types
        self.kernel_emb_dim = kernel_emb_dim
        self.cot_eps = cot_eps
        self.cot_iters = cot_iters
        self.cot_bandwidth = cot_bandwidth
        self.cot_alpha = cot_alpha
        self.cot_beta = cot_beta

        padding = stride
        if patch_len > configs.pred_len:
            patch_len = configs.pred_len

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Forecast Head (residual branch)
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

        # ===== TICaL plugin =====
        if self.use_text and self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Text -> temporal kernels
            self.text2k = Text2Kernels(self.txt_dim, K=self.K, shape_types=self.shape_types, H_scale=kernel_H_scale)
            # Kernel params -> embedding (for COT/contrast)
            self.kemb = KernelEmbed(K=self.K, S=self.shape_types, dim=self.kernel_emb_dim)
            # TIR 1->D expansion
            self.tir_expand = nn.Linear(1, self.enc_in, bias=True)
            nn.init.xavier_uniform_(self.tir_expand.weight)
            nn.init.zeros_(self.tir_expand.bias)
            # Optional per-variable hint (kept for compatibility; not strictly needed here)
            self.tir_var_proj = nn.Parameter(torch.ones(self.enc_in))
            # Per-variable gating (input: TS global + text mean + var hint)
            gate_in = configs.d_model + self.txt_dim + self.enc_in
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, 256), nn.ReLU(),
                nn.Linear(256, self.enc_in)
            )

            # Future delta projection for auxiliary objectives
            self.future_proj = FutureDeltaProj(in_dim=3*configs.enc_in, out_dim=kernel_emb_dim)
            for p in self.future_proj.parameters():
                p.requires_grad = not freeze_future_proj

        # Loss weights
        self.lmb_mse = 1.0
        self.lmb_cot = lmb_cot
        self.lmb_delta = lmb_delta
        self.lmb_entropy = lmb_entropy
        self.lmb_tv = lmb_tv

    # ---------------- Residual branch ----------------
    def _encode_and_predict(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_enc: [B, L, D]
        return:
          y_res: [B, H, D]
          ts_global: [B, d_model]  (for gating)
        """
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # patching & encode
        x = x.permute(0, 2, 1)                       # [B,D,L]
        enc_out, n_vars = self.patch_embedding(x)    # enc_out: [B*D, P, d_model]
        enc_out, _ = self.encoder(enc_out)           # [B*D, P, d_model]

        # reshape & head
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])  # [B,D,P,d_model]
        enc_out_perm = enc_out.permute(0, 1, 3, 2)                                 # [B,D,d_model,P]
        y_res = self.head(enc_out_perm).permute(0, 2, 1)                           # [B,H,D]

        # de-normalize
        H = y_res.size(1)
        y_res = y_res * (stdev[:, 0, :].unsqueeze(1).repeat(1, H, 1))
        y_res = y_res + (means[:, 0, :].unsqueeze(1).repeat(1, H, 1))

        # TS global for gating: mean over D,P -> d_model
        ts_global = enc_out.mean(dim=(1, 2))  # [B, d_model]
        return y_res, ts_global

    # ---------------- Text-induced response (TIR) ----------------
    def _text_induced_response(self, text_emb: torch.Tensor, H: int):
        """
        text_emb: [B,txt_dim] or [B,Q,txt_dim]
        return:
          y_tir_scalar: [B,H]  scalar response per time step
          kappa: [B,K,H], params: dict
        """
        kappa, params = self.text2k(text_emb, H)  # [B,K,H]
        if kappa is None:
            return None, None, None
        y_tir_scalar = kappa.sum(dim=1)  # [B,H]
        return y_tir_scalar, kappa, params

    # ----------------- Fusion -----------------
    def _fuse(self, y_res: torch.Tensor, y_tir_scalar: torch.Tensor,
              ts_global: torch.Tensor, text_emb: torch.Tensor):
        """
        y_res: [B,H,D], y_tir_scalar: [B,H], ts_global: [B, d_model]
        text_emb: [B,txt_dim] or [B,Q,txt_dim]
        """
        if y_tir_scalar is None:
            return y_res, None

        B, H, D = y_res.shape
        # Expand TIR to D
        y_tir = self.tir_expand(y_tir_scalar.unsqueeze(-1))  # [B,H,1] -> [B,H,D]
        # Optional additional per-variable scaling
        y_tir = y_tir * self.tir_var_proj[None, None, :]

        # Aggregate/mean text if multiple
        if text_emb.dim() == 3:
            e_mean = text_emb.mean(dim=1)  # [B,txt_dim]
        else:
            e_mean = text_emb             # [B,txt_dim]

        var_hint = self.tir_var_proj[None, ...]              # [1,D]
        gate_in_all = torch.cat([ts_global, e_mean, var_hint.expand(B, -1)], dim=-1)  # [B,dmodel+txt+D]
        gate = torch.sigmoid(self.gate_mlp(gate_in_all))     # [B,D]
        gate = 0.00001*gate.view(B, self.enc_in, D)                            # broadcast over time

        # Numerically stable convex mixing
        y_hat = y_res + gate * (y_tir - y_res)
        return y_hat, gate

    # --------------- Aux losses: COT + Delta-Contrastive --------------
    def _aux_losses(self, kappa: torch.Tensor, params: Dict[str, torch.Tensor],
                    y_future: torch.Tensor, H: int):
        if (kappa is None) or (y_future is None):
            return {}

        # Kernel params -> embedding
        k_emb = self.kemb(params)  # [B,K,Dk]
        # Future delta token representation
        with torch.set_grad_enabled(self.future_proj.proj.weight.requires_grad):
            f_pos = self.future_proj(y_future)  # [B,H,Dk]

        # Cost & bandwidth
        C = causal_cost(k_emb, f_pos, params['mu'], alpha=self.cot_alpha, beta=self.cot_beta)  # [B,K,H]
        if self.cot_bandwidth is not None and self.cot_bandwidth > 0:
            B, K, Hh = C.shape
            idx = torch.arange(Hh, device=C.device).float()[None, None, :] + 1.0
            mu = params['mu'][..., None]
            # Adaptive bandwidth: max(user_bandwidth, mean sigma)
            bw = max(float(self.cot_bandwidth), float(params['sigma'].mean().detach().clamp_min(1.0)))
            band = (torch.abs(idx - mu) <= bw).float()
        else:
            band = None

        # OT marginals aligned to kernel mass
        with torch.no_grad():
            r = (kappa.sum(-1) + 1e-6)                # [B,K]
            r = r / r.sum(dim=-1, keepdim=True)
            c = torch.full((C.size(0), C.size(2)), 1.0/C.size(2), device=C.device, dtype=C.dtype)

        Pi = sinkhorn(C, eps=self.cot_eps, iters=self.cot_iters, band_mask=band, r=r, c=c)  # [B,K,H]

        # Delta-Contrastive with negatives
        with torch.no_grad():
            shift = max(1, H // 6)
            f_neg1 = torch.roll(f_pos, shifts=shift, dims=1)
            perm = torch.randperm(f_pos.size(-1), device=f_pos.device)
            f_neg2 = f_pos[..., perm]

        loss_delta = delta_contrastive(k_emb, f_pos, Pi, negatives=[f_neg1, f_neg2], tau=0.07)
        # OT alignment residual (expected transport cost)
        loss_cot = (C * Pi).sum(dim=(1,2)).mean()

        # Entropy & smoothness regularizers
        shape_w = params['shape_w']  # [B,K,S]
        ent = -(shape_w.clamp_min(1e-8) * shape_w.clamp_min(1e-8).log()).sum(dim=-1).mean()
        mu = params['mu']            # [B,K]
        tv = (mu[:, 1:] - mu[:, :-1]).abs().mean()

        return {
            'loss_cot': loss_cot,
            'loss_delta': loss_delta,
            'loss_entropy': ent,
            'loss_tv': tv,
            'Pi': Pi.detach()
        }

    # -------------------- Task interfaces --------------------
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 text_emb=None, y_future=None, compute_aux=False):
        y_res, ts_global = self._encode_and_predict(x_enc)  # [B,H,D], [B,dmodel]
        H = y_res.size(1)

        if not (self.use_text and text_emb is not None):
            return y_res, {}

        # Text-induced response
        y_tir_scalar, kappa, params = self._text_induced_response(text_emb, H)
        y_hat, gate = self._fuse(y_res, y_tir_scalar, ts_global, text_emb)

        aux = {}
        if compute_aux and (y_future is not None):
            aux = self._aux_losses(kappa, params, y_future, H)
        aux.update({
            'gate': gate.detach() if gate is not None else None,
            'kappa': kappa.detach() if kappa is not None else None,
            'kernel_params': {k: v.detach() for k, v in params.items()} if params is not None else None
        })
        return y_hat, aux

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    # 兼容旧 forward，同时支持新参数
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, text_emb=None, y_future=None, compute_aux=False):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            y_hat, aux = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                       text_emb=text_emb, y_future=y_future, compute_aux=compute_aux)
            return y_hat[:, -self.pred_len:, :], aux  # [B,pred_len,D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
