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
        self.emb_scale = math.sqrt(emb_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, enc_input, input_key_mask):
        in_embedded = self.pe(self.emb(enc_input))
        encoded = self.enc(in_embedded.transpose(1, 0), src_key_padding_mask=input_key_mask)
        return encoded

    def decode(self, encoded, input_key_mask, dec_input, output_key_mask, out_offsets):
        out_embedded = self.pe(self.emb(dec_input), out_offsets)
        dec_mask = self._generate_square_subsequent_mask(dec_input.shape[1]).to(dec_input.device)
        decoded = self.dec(out_embedded.transpose(1, 0), encoded, dec_mask, tgt_key_padding_mask=output_key_mask,
                           memory_key_padding_mask=input_key_mask)
        return self.lin(decoded).transpose(1, 0)

    def forward(self, enc_input, dec_input, input_key_mask, output_key_mask, out_offsets):
        encoded = self.encode(enc_input, input_key_mask)
        return self.decode(encoded, input_key_mask, dec_input, output_key_mask, out_offsets)

    def beam_decode(self,
                    input: torch.Tensor,
                    output_seed: torch.Tensor,
                    beam_width: int = 5,
                    max_len: int = 175,
                    end_token: int = -1,
                    position_offset: int = 0):
        if input.ndim == 1:
            input = input.unsqueeze(0)
        # if output_seed.ndim == 1:
        #     output_seed = output_seed.unsqueeze(0)

        encoded_input = self.encode(input, None)

        candidates = [(0, output_seed.tolist())]
        for pi in range(max_len):
            potential_candidates = []
            for prob, candidate in candidates:
                offset = torch.arange(position_offset, position_offset+len(candidate))
                t_candidate = torch.tensor(candidate, dtype=torch.long).unsqueeze(0)
                decoded = self.decode(encoded_input, None, t_candidate, None, offset)
                probs, indices = nn.functional.log_softmax(decoded, dim=-1).sort(dim=-1, descending=True)
                for p, vi in zip(probs[0, -1, :beam_width], indices[0, -1, :beam_width]):
                    potential_candidates.append((p.item() + prob, candidate + [vi.item()]))

            candidates.clear()
            potential_candidates = sorted(potential_candidates, key=lambda x: x[0], reverse=True)
            for i in range(beam_width):
                if potential_candidates[i][1][-1] == end_token:
                    return potential_candidates[i][1]
                candidates.append(potential_candidates[i])

        return candidates[0][1]