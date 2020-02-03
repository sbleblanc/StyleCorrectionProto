import tarfile
import io
import math
import itertools as it
import numpy as np
from collections import Counter
from typing import List, Callable, Tuple
import torch
import torch.nn as nn
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

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def pretrain_batch_gen(data, mask_idx, pad_idx, vocab_range, bs=32):
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
            pe_offsets.append(mask_start)
            masked_input = vector.clone()
            for oi, (mii, a) in enumerate(zip(range(mask_start, mask_start + mask_len), masking_probs.multinomial(mask_len, replacement=True))):
                if a == 0:
                    masked_input[mii] = mask_idx
                elif a == 1:
                    masked_input[mii] = torch.randint(vocab_range[0], vocab_range[1], [1])
            masked_enc_input.append(masked_input)
            output.append(vector[mask_start:mask_start+mask_len])

        if (i+1) % bs == 0 or i == len(data) - 1:
            m_input = torch.empty([bs, longest_input], dtype=torch.int).fill_(pad_idx)
            m_input_key_mask = torch.zeros([bs, longest_input]).bool()
            m_output = torch.empty([bs, longest_input], dtype=torch.int).fill_(pad_idx)
            m_output_key_mask = torch.zeros([bs, longest_input]).bool()
            for i in range(len(masked_enc_input)):
                in_v = masked_enc_input[i]
                out_v = output[i]
                m_input[i, :in_v.shape[0]] = in_v
                m_input_key_mask[i, in_v.shape[0]:] = True
                m_output[i, :out_v.shape[0]] = out_v
                m_output_key_mask[i, out_v.shape[0]:] = True
            yield m_input, m_input_key_mask, m_output, m_output_key_mask, torch.tensor(pe_offsets)
            longest_input = 0
            longest_output = 0
            masked_enc_input.clear()
            output.clear()
            pe_offsets.clear()


unk_token = "<unk>"
mask_token = "<mask>"
pad_token = "<pad>"

bcl = BookCorpusLoader(tokenize=lambda x: x.strip().split(' '), topk=10)
dataset = bcl.extract_from_archive("temp/datasets/BookCorpus_unique.tar.gz")
vocab_count = Counter(it.chain(*dataset))
vocab = [mask_token, pad_token, unk_token] + [w for w, _ in it.takewhile(lambda x: x[1] > 5, vocab_count.most_common())]
vocab_set = set(vocab)
wtoi = dict([(w, i) for i, w in enumerate(vocab)])
num_dataset = []

for s in dataset:
    num_dataset.append(torch.tensor([wtoi[w] if w in vocab_set else wtoi[unk_token] for w in s], dtype=torch.int))

pretrain_batch_gen(num_dataset, wtoi[mask_token], wtoi[pad_token], (3, len(vocab)))

tel = nn.TransformerEncoderLayer(80, 8)
te = nn.TransformerEncoder(tel, 8, norm=mnorm.LayerNorm(80))

data = torch.rand([10,2,80])
print(te(data).shape)