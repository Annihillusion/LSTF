from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from traceback import print_tb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from random import randrange
from layers.Transformer_EncDec import gMLPDecoder, gMLPDecoderLayer
from layers.SelfAttention_Family import FullAttention, gMLPAttentionLayer, SplitAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer


def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, input_t):
        value, pos, x_mask = input_t
        out_val, out_pos, x_mask = self.fn((value, pos, x_mask))
        return ((out_val + value), (out_pos + pos), x_mask)


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
        causal = False,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
        circulant_matrix = False, 
        enc_in = 7
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

    def forward(self, value, pos, gate_res = None, x_mask = None):
        if x_mask is None:
            print('error')
            exit()
        device, n, h = value.device, value.shape[1], self.heads

        res, gate = value, pos
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias
        impute_bias = self.impute_bia

        # if self.circulant_matrix:
        #     # build the circulant matrix

        #     dim_seq = weight.shape[-1]
        #     weight = F.pad(weight, (0, dim_seq), value = 0)
        #     weight = repeat(weight, '... n -> ... (r n)', r = dim_seq)
        #     weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
        #     weight = weight[:, :, (dim_seq - 1):]

        #     # give circulant matrix absolute position awareness

        #     pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
        #     weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        # if self.causal:
        #     weight, bias = weight[:, :n, :n], bias[:, :n]
        #     mask = torch.ones(weight.shape[-2:], device = device).triu_(1).bool()
        #     mask = rearrange(mask, 'i j -> () i j')
        #     weight = weight.masked_fill(mask, 0.)
        
        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)
        
        rerange_mask = 1 - x_mask.unsqueeze(1).repeat(1,h,1,1)
        
        mask_bias = einsum('b h n d, h n d m -> b h n m', rerange_mask, impute_bias)
        
        mask_gate = einsum('b h n d, h m n -> b h m d', mask_bias, weight)
        # print('gate shape : {}'.format(gate.shape))
        # print('wieght shape : {}'.format(weight.shape))
        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        
        
        gate = gate + rearrange(bias, 'h n -> () h n ()') + mask_gate 

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res, gate, x_mask


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
        heads = 1,
        attn_dim = None,
        causal = False,
        act = nn.Identity(),
        circulant_matrix = False, 
        enc_in = 7
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
        
        self.attn = AutoCorrelationLayer(AutoCorrelation(False, 3, 0.1),dim, heads) if exists(attn_dim) else None
        self.attn_projection = nn.Linear(dim, dim_ff)
        self.batchnormlayer1 = torch.nn.LayerNorm(dim_ff)

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix = circulant_matrix, enc_in = enc_in)
        self.proj_out_value = nn.Linear(dim_ff , dim)
        self.proj_out_pos = nn.Linear(dim_ff , dim)
    def forward(self, input_t):
        value, pos, x_mask = input_t
        if exists(self.attn):
            gate_res, _ = self.attn(pos, pos, pos, x_mask)
            gate_res = self.attn_projection(gate_res)
        else:
            gate_res =  None
        value = self.proj_in_value(value)
        pos = self.proj_in_pos(pos)
        value, pos, x_mask = self.sgu(value, pos, gate_res = None, x_mask = x_mask)
        value = self.proj_out_value(value)
        pos = self.proj_out_pos(pos)
        return value, pos, x_mask

class gMLP_split(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, out_len, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.0, embed='fixed', freq='h', activation='gelu', device=torch.device('cuda:0'), 
                prob_survival = 1, impute = False, factor = 5, split = True, pos_val_type = 0, attn_dim = None, kernel_size = 25):
        super(gMLP_split, self).__init__()
        self.pred_len = out_len
        self.label_len = 48
        self.impute = impute
        self.split = split
        self.seq_len = seq_len
        self.pos_val_type = pos_val_type
        print('pos_val_type : {}'.format(pos_val_type))
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, split)
        self.activation = F.relu if activation == "relu" else F.gelu
        # Encoder
        self.layers = nn.ModuleList(
            [
                Residual(
                    PreNorm(d_model, 
                            gMLPBlock(dim = d_model, heads = n_heads, dim_ff = d_ff, seq_len = seq_len, act = self.activation, attn_dim = attn_dim, enc_in = enc_in)
                    )
                ) 
                for i in range(e_layers)
            ])
        # Decoder
        self.dec_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, split)
        self.prob_survival = prob_survival
        print("prob survival:{}".format(self.prob_survival))
        self.decoder = gMLPDecoder(
            [
                gMLPDecoderLayer(
                    gMLPBlock(dim = d_model, heads = n_heads, dim_ff = d_ff, seq_len = self.pred_len + self.label_len,  act = self.activation, attn_dim = attn_dim, enc_in = enc_in),
                    gMLPAttentionLayer(SplitAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False, split = True),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model), 
            split = True
        )
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, c_out)
        ) 

        self.value_embedding = nn.Linear(enc_in, d_model)
        self.LayerNorm_val = nn.LayerNorm(d_model)

        print("seq_len : {}".format(seq_len))
        self.decomp = series_decomp(kernel_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, x_mask=None):

        x_mask = x_mask.to(x_enc.device).float()
        B, S, D = x_enc.shape
        mask_y = torch.ones((B, self.pred_len, D)).float().to(x_enc.device)
        mask_y = torch.cat([x_mask[:, -self.label_len:, :], mask_y], dim=1)
        # x_mean = (torch.sum(x_enc, axis = 1) / torch.sum(x_mask, axis = 1)).unsqueeze(-2).repeat(1,self.pred_len,1)
        # x_mask = x_mask.unsqueeze(-1)
        
        # AutoFormer
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init, trend_init = self.decomp(x_enc)
        # trend_init = torch.cat([trend_init[:, -self.pred_len:, :], x_mean], dim=1)
        # seasonal_init = torch.cat([seasonal_init[:, -self.pred_len:, :], zeros], dim=1)
        # print('mask_total : {}'.format(torch.sum(x_mask)))
        
        # val_embedding = torch.sum(self.value_embedding(x_enc.unsqueeze(-1)) * x_mask, axis = -2) / torch.sum(x_mask, axis = -2) 
        # val_embedding = self.LayerNorm_val(val_embedding)

        # val_embedding = self.value_embedding(x_enc) / torch.sum(x_mask, axis = -1).unsqueeze(-1)
        # val_embedding = self.LayerNorm_val(val_embedding)
        
        # x_enc = trend_init
        # x_dec = seasonal_init
        
        B, seq_len ,dim = x_enc.shape
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)

        enc_value_embedding, enc_pos_embedding = self.enc_embedding(x_enc, x_mark_enc)
        
        if self.pos_val_type == 0:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding, enc_pos_embedding
        elif self.pos_val_type == 1:
            enc_pos_embedding, enc_value_embedding = enc_value_embedding, enc_pos_embedding
        elif self.pos_val_type == 2:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding + enc_pos_embedding, enc_pos_embedding
        elif self.pos_val_type == 3:
            enc_value_embedding, enc_pos_embedding = enc_value_embedding , enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 4:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding + enc_value_embedding , enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 5:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding , enc_pos_embedding + enc_value_embedding 
            
        
        dec_value_embedding, dec_pos_embedding = self.dec_embedding(x_dec, x_mark_dec)
        enc_out_value, enc_out_pos, x_mask = nn.Sequential(*layers)((enc_value_embedding, enc_pos_embedding, x_mask))
        dec_out = self.decoder(dec_value_embedding, enc_out_value, x_p = dec_pos_embedding, cross_p = enc_out_pos, x_mask = mask_y)

        out = self.to_logits(dec_out)
        
        
        if self.impute:
            return out
        else:
            return out[:,-self.pred_len:,:] 