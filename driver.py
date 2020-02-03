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


class BookCorpusLoader(object):

    def __init__(self,
                 tokenize: Callable[[str], List[str]],
                 topk: float = float('inf'),
                 preprocess: Callable[[str], str] = None,
                 verbose: bool = False):
        self.tokenize = tokenize
        self.topk = topk
        self.preprocess = preprocess
        self.verbose = verbose

    def __process_lines(self, raw_lines: List[str]) -> List[str]:
        tok_pre_sentences = []
        for line in raw_lines:
            if self.preprocess:
                pre_sentence = self.preprocess(line.strip())
            else:
                pre_sentence = line.strip()
            tok_pre_sentences.append(self.tokenize(pre_sentence))
        return tok_pre_sentences
        # return list(it.chain(*tok_pre_sentences))

    def extract_from_archive(self, corpus_tar_gz: str) -> List[List[str]]:
        processed_book_sentences = []
        with tarfile.open(corpus_tar_gz, 'r:gz') as tar_file:
            books = tar_file.getmembers()
            if self.topk != float('inf'):
                selector = np.zeros(len(books))
                chosen_books = np.random.choice(len(books), size=[self.topk], replace=False)
                selector[chosen_books] = 1
                books = list(it.compress(books, selector))
            print('Processing books...', end='')
            for i, b in enumerate(books):
                reader = io.TextIOWrapper(tar_file.extractfile(b))
                raw_text = reader.read(None).splitlines()
                processed_book_sentences.extend(self.__process_lines(raw_text))
            print('DONE')

        return processed_book_sentences

    def extract_from_text(self, corpus_fn: str) -> Tuple[List[List[str]], List[List[str]]]:
        raw_lines = []
        with open(corpus_fn, 'r') as in_file:
            for line in in_file:
                if len(raw_lines) > self.topk:
                    break
                raw_lines.append(line)
        num_valid_lines = int(len(raw_lines) * (self.valid_split_ratio * 100) // 100)
        processed_train_lines = self.__process_lines(raw_lines[:num_valid_lines])
        processed_valid_lines = self.__process_lines(raw_lines[num_valid_lines:])
        return [processed_train_lines], [processed_valid_lines]


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


def pretrain_batch_gen(data, mask_idx, pad_idx, vocab_range, bs=32, device="cpu"):
    masking_probs = torch.tensor([0.8, 0.1, 0.1])
    masked_enc_input = []
    output = []
    pe_offsets = []
    longest_input = 0
    longest_output = 0
    for i, vector in enumerate(data):
        if vector.shape[0] > longest_input:
            longest_input = vector.shape[0]
        mask_len = vector.shape[0] // 2
        if mask_len > longest_output:
            longest_output = mask_len
        if mask_len > 0:
            mask_start = torch.randint(0, mask_len, [1])
            pe_offsets.append(mask_start.item())
            masked_input = vector.clone()
            for oi, (mii, a) in enumerate(zip(range(mask_start, mask_start + mask_len), masking_probs.multinomial(mask_len, replacement=True))):
                if a == 0:
                    masked_input[mii] = mask_idx
                elif a == 1:
                    masked_input[mii] = torch.randint(vocab_range[0], vocab_range[1], [1])
            masked_enc_input.append(masked_input)
            output.append(vector[mask_start:mask_start+mask_len])

        if (i+1) % bs == 0 or i == len(data) - 1:
            m_input = torch.empty([bs, longest_input], dtype=torch.long).fill_(pad_idx).to(device)
            m_input_key_mask = torch.zeros([bs, longest_input]).bool().to(device)
            m_output = torch.empty([bs, longest_output], dtype=torch.long).fill_(pad_idx).to(device)
            m_output_key_mask = torch.zeros([bs, longest_output]).bool().to(device)
            offsets = torch.zeros([bs, longest_output], dtype=torch.long).to(device)
            for i in range(len(masked_enc_input)):
                in_v = masked_enc_input[i]
                out_v = output[i]
                m_input[i, :in_v.shape[0]] = in_v
                m_input_key_mask[i, in_v.shape[0]:] = True
                m_output[i, :out_v.shape[0]] = out_v
                m_output_key_mask[i, out_v.shape[0]:] = True
                offsets[i] = torch.arange(pe_offsets[i], pe_offsets[i] + longest_output)
            shifted_ouputs = torch.empty([bs, longest_output], dtype=torch.long).fill_(mask_idx).to(device)
            shifted_ouputs[:, 1:] = m_output[:, :-1]
            yield m_input, m_input_key_mask, m_output, shifted_ouputs, m_output_key_mask, offsets
            longest_input = 0
            longest_output = 0
            masked_enc_input.clear()
            output.clear()
            pe_offsets.clear()

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
