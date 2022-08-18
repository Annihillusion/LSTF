import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


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
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
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
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
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
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, x_mask=None):
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
            enc_value_embedding, enc_pos_embedding = enc_value_embedding , enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 4:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding + enc_value_embedding , enc_pos_embedding + enc_value_embedding
        elif self.pos_val_type == 5:
            enc_value_embedding, enc_pos_embedding = enc_pos_embedding , enc_pos_embedding + enc_value_embedding

        dec_value_embedding, dec_pos_embedding = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out_value, enc_out_pos, x_mask = self.encoder((enc_value_embedding, enc_pos_embedding, x_mask))
        # dec
        seasonal_part, trend_part = dec_out = self.decoder(dec_value_embedding, enc_out_value, x_p=dec_pos_embedding,
                                                           cross_p=enc_out_pos, x_mask=mask_y)
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
