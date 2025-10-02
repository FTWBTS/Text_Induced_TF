import torch
import torch.nn as nn
from layers.Embed_TimeBridge import PatchEmbed
from layers.SelfAttention_Family_TimeBridge import TSMixer, ResAttention
from layers.Transformer_EncDec_TimeBridge import TSEncoder, IntAttention, PatchSampling, CointAttention
from layers.TICAL import Text2Kernels, KernelEmbed, FutureDeltaProj, causal_cost, sinkhorn, delta_contrastive
from layers.TICALPlugin import TICALPlugin

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.tical_plugin = TICALPlugin(configs)
        self.use_text = configs.use_text

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,text_emb, y_future, compute_aux):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc_init = x_enc
        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)

        if self.use_text and text_emb is not None:
            y_tir_scalar, kappa, params = self.tical_plugin._text_induced_response(text_emb, dec_out.size(1))
            
            aux = {}
            if compute_aux and y_future is not None:
                aux = self.tical_plugin._aux_losses(kappa, params, y_future, dec_out.size(1))
            
            y_hat, gate = self.tical_plugin._fuse(dec_out, y_tir_scalar, x_enc_init, text_emb)
            aux.update({
                'gate': gate.detach() if gate is not None else None,
                'kappa': kappa.detach() if kappa is not None else None,
                'kernel_params': {k: v.detach() for k, v in params.items()} if params is not None else None
            })
            return y_hat, aux
        
        return dec_out * std + mean, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,mask=None, text_emb: torch.Tensor = None,y_future: torch.Tensor = None, compute_aux: bool = False):
        y_hat, aux = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,text_emb=text_emb, y_future=y_future, compute_aux=compute_aux)
        return y_hat[:, -self.pred_len:, :], aux  # [B,pred_len,D]
