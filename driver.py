import tarfile
import io
import math
import itertools as it
import numpy as np
from collections import Counter
from typing import List, Callable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.modules.normalization as mnorm
from stylecorrection.loaders.bookcorpus import BookCorpusLoader, PretrainingDataset


bl = BookCorpusLoader(lambda x: x.strip().split(' '), 100, 10000)
bl.extract_from_archive('temp/datasets/BookCorpus_unique.tar.gz')
pds = PretrainingDataset(bl)
b = next(pds())


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x, offsets=None):
        if offsets is not None:
            x = x + self.pe.squeeze(0)[offsets]
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class s2s(nn.Module):

    def __init__(self,
                 num_emb: int,
                 emb_dim: int,
                 nhead: int = 8,
                 ff_dim: int = 2048,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6,
                 device="cpu"):
        super(s2s, self).__init__()

        self.emb = nn.Embedding(num_emb, emb_dim)
        self.pe = PositionalEncoding(emb_dim, device=device)
        l_norm = nn.LayerNorm(emb_dim)
        tel = nn.TransformerEncoderLayer(emb_dim, nhead, ff_dim)
        tdl = nn.TransformerDecoderLayer(emb_dim, nhead, ff_dim)
        self.enc = nn.TransformerEncoder(tel, num_enc_layers, norm=l_norm)
        self.dec = nn.TransformerDecoder(tdl, num_dec_layers, norm=l_norm)
        self.lin = nn.Linear(emb_dim, num_emb)

    def forward(self, enc_input, dec_input, input_key_mask, output_key_mask, out_offsets):
        in_embedded = self.pe(self.emb(enc_input))
        encoded = self.enc(in_embedded.transpose(1, 0), src_key_padding_mask=input_key_mask)
        out_embedded = self.pe(self.emb(dec_input), out_offsets)
        decoded = self.dec(out_embedded.transpose(1, 0), encoded, torch.ones(dec_input.shape[1], dec_input.shape[1]).tril(), tgt_key_padding_mask=output_key_mask, memory_key_padding_mask=input_key_mask)
        return self.lin(decoded).transpose(1, 0)




device = "cuda:0" if torch.cuda.is_available() else "cpu"

unk_token = "<unk>"
mask_token = "<mask>"
pad_token = "<pad>"
bos_token = "<bos>"
eos_token = "<eos>"

bcl = BookCorpusLoader(tokenize=lambda x: x.strip().split(' '), topk=1)
dataset = bcl.extract_from_archive("temp/datasets/BookCorpus_unique.tar.gz")
vocab_count = Counter(it.chain(*dataset))
vocab = [mask_token, pad_token, unk_token, bos_token, eos_token] + [w for w, _ in it.takewhile(lambda x: x[1] > 5, vocab_count.most_common())]
vocab_set = set(vocab)
wtoi = dict([(w, i) for i, w in enumerate(vocab)])
num_dataset = []
model = s2s(len(vocab), 200).to(device)

for s in dataset:
    num_dataset.append(torch.tensor([wtoi[bos_token]] + [wtoi[w] if w in vocab_set else wtoi[unk_token] for w in s] + [wtoi[eos_token]], dtype=torch.int))

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=wtoi[pad_token]).to(device)

for i in range(100):
    train_losses = []
    for enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets in pretrain_batch_gen(num_dataset,
                                                                                                 wtoi[mask_token],
                                                                                                 wtoi[pad_token],
                                                                                                 (3, len(vocab))):
        optimizer.zero_grad()
        out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
        loss = criterion(out.contiguous().view(-1, len(vocab)), dec_out.view(-1))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    print('Iteration {} : {:.4f}'.format(i, torch.tensor(train_losses).mean()))
