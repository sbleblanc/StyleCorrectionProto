import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, offsets=None):
        if offsets is not None:
            x = x + self.pe.squeeze(0)[offsets]
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerS2S(nn.Module):

    def __init__(self,
                 num_emb: int,
                 emb_dim: int,
                 nhead: int = 8,
                 ff_dim: int = 2048,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6):
        super(TransformerS2S, self).__init__()

        self.emb = nn.Embedding(num_emb, emb_dim)
        self.pe = PositionalEncoding(emb_dim)
        l_norm = nn.LayerNorm(emb_dim)
        tel = nn.TransformerEncoderLayer(emb_dim, nhead, ff_dim)
        tdl = nn.TransformerDecoderLayer(emb_dim, nhead, ff_dim)
        self.enc = nn.TransformerEncoder(tel, num_enc_layers, norm=l_norm)
        self.dec = nn.TransformerDecoder(tdl, num_dec_layers, norm=l_norm)
        self.lin = nn.Linear(emb_dim, num_emb)

    def forward(self, enc_input, dec_input, input_key_mask, output_key_mask, out_offsets, dec_mask):
        in_embedded = self.pe(self.emb(enc_input))
        encoded = self.enc(in_embedded.transpose(1, 0), src_key_padding_mask=input_key_mask)
        out_embedded = self.pe(self.emb(dec_input), out_offsets)
        decoded = self.dec(out_embedded.transpose(1, 0), encoded, dec_mask, tgt_key_padding_mask=output_key_mask, memory_key_padding_mask=input_key_mask)
        return self.lin(decoded).transpose(1, 0)