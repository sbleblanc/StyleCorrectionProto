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

    def beam_decode_2(self,
                      input: torch.Tensor,
                      output_seed: torch.Tensor,
                      beam_width: int = 5,
                      max_len: int = 175,
                      end_token: int = -1,
                      noising_beta: float = 0.,
                      topmost_noising: bool = False,
                      temperature: float = 1.,
                      top_only: bool = True,
                      device: str = 'cpu'):
        if input.ndim == 1:
            input = input.unsqueeze(0)

        completed_beams = []
        encoded_input = self.encode(input, None)
        scores = torch.zeros(1).to(device)
        candidates = output_seed.unsqueeze(0)
        with torch.no_grad():
            for pi in range(max_len):
                decoded = self.decode(encoded_input.repeat_interleave(candidates.shape[0], 1), None, candidates, None, None)
                probs, indices = nn.functional.log_softmax(decoded/temperature, dim=-1).sort(dim=-1, descending=True)
                probs = probs[:, -1, :beam_width]
                indices = indices[:, -1, :beam_width]
                temp_scores = scores.repeat_interleave(beam_width).view(-1, beam_width) + probs
                if noising_beta > 0.:
                    noise = torch.rand_like(temp_scores) * noising_beta
                    temp_scores += noise
                    # temp_scores = (temp_scores.exp() - noise).max(torch.tensor([0.]).to(device)).log()
                # logits, indices = decoded.sort(dim=-1, descending=True)
                # probs = nn.functional.log_softmax(logits[:, -1, :beam_width], dim=-1)
                # indices = indices[:, -1, :beam_width]
                # if noising_beta > 0.:
                #     noise = torch.rand_like(probs) * noising_beta
                #     if topmost_noising:
                #         probs[:, 0] = (probs.exp()[:, 0] - noise[:, 0]).max(torch.tensor([0.]).to(device)).log()
                #     else:
                #         probs = (probs.exp() - noise).max(torch.tensor([0.]).to(device)).log()
                # temp_scores = scores.repeat_interleave(beam_width).view(-1, beam_width) + probs
                final_scores, final_indices = temp_scores.view(-1).sort(descending=True)
                scores = scores.repeat_interleave(beam_width).view(-1)[final_indices[:beam_width]] + probs.contiguous().view(-1)[final_indices[:beam_width]]
                # scores = final_scores[:beam_width]
                candidates_previous = final_indices[:beam_width] // beam_width
                new_candidates = torch.zeros(beam_width, pi+2, dtype=torch.long).to(device)
                new_candidates[:, :-1] = candidates[candidates_previous]
                new_candidates[:, -1] = indices.contiguous().view(-1)[final_indices[:beam_width]]
                potential_finished = new_candidates[:, -1] == end_token
                finished = potential_finished.nonzero().view(-1)
                unfinished = (~potential_finished).nonzero().view(-1)
                candidates = new_candidates[unfinished]
                for idx in finished:
                    completed_beams.append((scores[idx].item(), new_candidates[idx].cpu()))
                    if len(completed_beams) == beam_width:
                        break
                scores = scores[unfinished]
                if candidates.ndim == 0 or len(completed_beams) == beam_width:
                    break

        if len(completed_beams) != beam_width:
            for i in range(beam_width - len(completed_beams)):
                completed_beams.append((scores[i].item(), candidates[i]))

        completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
        return [b for s, b in completed_beams]


    def beam_decode(self,
                    input: torch.Tensor,
                    output_seed: torch.Tensor,
                    beam_width: int = 5,
                    max_len: int = 175,
                    end_token: int = -1,
                    noising_beta: float = 0.,
                    top_only: bool = True,
                    device: str = 'cpu'):
        if input.ndim == 1:
            input = input.unsqueeze(0)
        # if output_seed.ndim == 1:
        #     output_seed = output_seed.unsqueeze(0)

        encoded_input = self.encode(input, None)

        candidates = [(0, output_seed.tolist())]
        final_candidates = []
        for pi in range(max_len):
            potential_candidates = []
            for prob, candidate in candidates:
                # offset = torch.arange(position_offset, position_offset+len(candidate))
                t_candidate = torch.tensor(candidate, dtype=torch.long).unsqueeze(0).to(device)
                decoded = self.decode(encoded_input, None, t_candidate, None, None)
                probs, indices = nn.functional.log_softmax(decoded, dim=-1).sort(dim=-1, descending=True)
                probs = probs[0, -1, :beam_width]
                indices = indices[0, -1, :beam_width]
                if noising_beta > 0.:
                    noise = torch.rand_like(probs) * noising_beta
                    probs = (probs.exp() + noise).log()
                    probs, resorted_indices = probs.sort(dim=-1, descending=True)
                    indices = indices[resorted_indices]
                for p, vi in zip(probs, indices):
                    potential_candidates.append((p.item() + prob, candidate + [vi.item()]))

            candidates.clear()
            potential_candidates = sorted(potential_candidates, key=lambda x: x[0], reverse=True)
            for i in range(beam_width):
                if potential_candidates[i][1][-1] == end_token:
                    final_candidates.append(potential_candidates[i])
                    beam_width -= 1
                else:
                    candidates.append(potential_candidates[i])
            if beam_width == 0:
                break

        if beam_width > 0:
            final_candidates.extend(candidates)

        final_candidates = sorted(final_candidates, key=lambda x: x[0], reverse=True)

        if top_only:
            return final_candidates[0][1]
        else:
            return [torch.tensor(t) for p, t in final_candidates]