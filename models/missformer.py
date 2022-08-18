import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from random import randrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import gMLPAttentionLayer, SplitAttention


def exists(val):
    return val is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, input_t):
        value, pos, x_mask = input_t
        out_val, out_pos, x_mask = self.fn((value, pos, x_mask))
        return (out_val + value), (out_pos + pos), x_mask


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_pos = nn.LayerNorm(dim)
        self.norm_value = nn.LayerNorm(dim)

    def forward(self, input_t, **kwargs):
        value, pos, x_mask = input_t
        value_out = self.norm_value(value)
        pos_out = self.norm_pos(pos)
        return self.fn((value_out, pos_out, x_mask), **kwargs)


class SpatialGatingUnit(nn.Module):
    def __init__(
            self,
            dim,
            dim_seq,
            causal=False,
            act=nn.Identity(),
            heads=1,
            init_eps=1e-3,
            circulant_matrix=False,
            enc_in=7
    ):
        super().__init__()
        dim_out = dim
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act

        # parameters

        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)

        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))
        self.impute_bia = nn.Parameter(torch.randn(heads, dim_seq, enc_in, dim // heads))

    def forward(self, value, pos, gate_res=None, x_mask=None):
        if x_mask is None:
            print('error')
            exit()
        device, n, h = value.device, value.shape[1], self.heads

        res, gate = value, pos
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias
        impute_bias = self.impute_bia

        gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)

        rerange_mask = 1 - x_mask.unsqueeze(1).repeat(1, h, 1, 1)

        mask_bias = einsum('b h n d, h n d m -> b h n m', rerange_mask, impute_bias)

        mask_gate = einsum('b h n d, h m n -> b h m d', mask_bias, weight)
        # print('gate shape : {}'.format(gate.shape))
        # print('wieght shape : {}'.format(weight.shape))
        gate = einsum('b h n d, h m n -> b h m d', gate, weight)

        gate = gate + rearrange(bias, 'h n -> () h n ()') + mask_gate

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return (self.act(gate) * res, gate, x_mask)


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class gMLPBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_ff,
            seq_len,
            heads=1,
            attn_dim=None,
            causal=False,
            act=nn.Identity(),
            circulant_matrix=False,
            enc_in=7
    ):
        super().__init__()
        self.proj_in_value = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        self.proj_in_pos = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = AutoCorrelationLayer(AutoCorrelation(False, 3, 0.1), dim, heads) if exists(attn_dim) else None
        self.attn_projection = nn.Linear(dim, dim_ff)
        self.batchnormlayer1 = torch.nn.LayerNorm(dim_ff)

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix,
                                     enc_in=enc_in)
        self.proj_out_value = nn.Linear(dim_ff, dim)
        self.proj_out_pos = nn.Linear(dim_ff, dim)

    def forward(self, input_t):
        value, pos, x_mask = input_t
        if exists(self.attn):
            gate_res, _ = self.attn(pos, pos, pos, x_mask)
            gate_res = self.attn_projection(gate_res)
        else:
            gate_res = None
        value = self.proj_in_value(value)
        pos = self.proj_in_pos(pos)
        value, pos, x_mask = self.sgu(value, pos, gate_res=None, x_mask=x_mask)
        value = self.proj_out_value(value)
        pos = self.proj_out_pos(pos)
        return value, pos, x_mask


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Residual(
                    #     PreNorm(configs.d_model,
                    #             gMLPBlock(dim=configs.d_model, heads=configs.n_heads, dim_ff=configs.d_ff,
                    #                       seq_len=configs.seq_len, act=self.activation, attn_dim=configs.attn_dim,
                    #                       enc_in=configs.enc_in)
                    #             )
                    # ),
                    gMLPBlock(dim=configs.d_model, heads=configs.n_heads, dim_ff=configs.d_ff,
                              seq_len=configs.seq_len, act=self.activation, attn_dim=configs.attn_dim,
                              enc_in=configs.enc_in),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    gMLPBlock(dim=configs.d_model, heads=configs.n_heads, dim_ff=configs.d_ff,
                              seq_len=self.pred_len + self.label_len, act=self.activation, attn_dim=configs.attn_dim,
                              enc_in=configs.enc_in),
                    gMLPAttentionLayer(
                        SplitAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads, mix=False, split=True
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, x_mask = None):
        # mask init
        x_mask = x_mask.to(x_enc.device).float()
        B, S, D = x_enc.shape
        mask_y = torch.ones((B, self.pred_len, D)).float().to(x_enc.device)
        mask_y = torch.cat([x_mask[:, -self.label_len:, :], mask_y], dim=1)
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc embedding
        enc_value_embedding, enc_pos_embedding = self.enc_embedding(x_enc, x_mark_enc)
        # from GMLP
        if self.pos_val_type == 0:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding, enc_pos_embedding
        elif self.pos_val_type == 1:
            enc_pos_embedding, enc_value_embedding = enc_value_embedding, enc_pos_embedding
        elif self.pos_val_type == 2:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding + enc_pos_embedding, enc_pos_embedding
        elif self.pos_val_type == 3:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding, enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 4:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding + enc_value_embedding, enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 5:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding, enc_pos_embedding + enc_value_embedding

        dec_value_embedding, dec_pos_embedding = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out_value, enc_out_pos, x_mask = self.encoder((enc_value_embedding, enc_pos_embedding, x_mask))
        # dec
        seasonal_part, trend_part = dec_out = self.decoder(dec_value_embedding, enc_out_value, x_p=dec_pos_embedding,
                                                           cross_p=enc_out_pos, x_mask=mask_y)
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
