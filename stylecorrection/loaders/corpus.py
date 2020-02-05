import tarfile
import io
import h5py
import torch
import numpy as np
import itertools as it
from collections import Counter
from typing import Callable, List, Tuple


class H5CorpusLoader(object):

    @classmethod
    def __process_lines(cls,
                        raw_lines: List[str],
                        tokenize: Callable[[str], List[str]],
                        preprocess: Callable[[str], str]) -> List[List[str]]:
        tok_pre_sentences = []
        for line in raw_lines:
            if preprocess:
                pre_sentence = preprocess(line.strip())
            else:
                pre_sentence = line.strip()
            tokenized = tokenize(pre_sentence)
            tok_pre_sentences.append(tokenized)
        return tok_pre_sentences

    @classmethod
    def create_from_compressed(cls,
                               h5_fn: str,
                               corpus_tar_gz: str,
                               tokenize: Callable[[str], List[str]],
                               preprocess: Callable[[str], str] = None):
        unk_token = "<unk>"
        mask_token = "<mask>"
        pad_token = "<pad>"
        bos_token = "<bos>"
        eos_token = "<eos>"

        vocab = [mask_token,
                 pad_token,
                 unk_token,
                 bos_token,
                 eos_token]
        wtoi = dict([(w, i) for i, w in enumerate(vocab)])
        vocab_counter = Counter()
        with tarfile.open(corpus_tar_gz, 'r:gz') as tar_file:
            books = tar_file.getmembers()[:100]
            print('Counting...', end='')
            line_count = 0
            longest = 0
            for i, b in enumerate(books):
                reader = io.TextIOWrapper(tar_file.extractfile(b))
                raw_text = reader.read(None).splitlines()
                processed_lines = cls.__process_lines(raw_text, tokenize, preprocess)
                for l in processed_lines:
                    for w in l:
                        if w not in wtoi:
                            wtoi[w] = len(vocab)
                            vocab.append(w)
                    if len(l) > longest:
                        longest = len(l)
                    vocab_counter.update(l)
                    line_count += 1
            print('DONE')

            dt = h5py.string_dtype(encoding='utf-8')
            with h5py.File(h5_fn, 'w') as h5_file:
                corpus_ds = h5_file.create_dataset('corpus', (line_count, longest), dtype='u4', compression='gzip', compression_opts=9)
                vocab_ds = h5_file.create_dataset('vocab', (len(vocab),), dtype=dt, compression='gzip', compression_opts=9)
                counts_ds = h5_file.create_dataset('vocab_counts', (len(vocab),), dtype=dt, compression='gzip', compression_opts=9)
                for i, w in enumerate(vocab):
                    vocab_ds[i] = w
                    counts_ds[i] = vocab_counter[w]

                line = 0
                for i, b in enumerate(books):
                    print('{:.2%}'.format(float(line)/line_count))
                    reader = io.TextIOWrapper(tar_file.extractfile(b))
                    raw_text = reader.read(None).splitlines()
                    processed_lines = cls.__process_lines(raw_text, tokenize, preprocess)
                    for l in processed_lines:
                        converted_line = [wtoi[w] for w in l]
                        corpus_ds[line, :len(l)] = converted_line
                        line += 1




class CorpusLoader(object):

    def __init__(self,
                 tokenize: Callable[[str], List[str]],
                 topk: float = float('inf'),
                 vocab_topk: int = 0,
                 min_freq: int = 2,
                 max_len: int = 256,
                 valid_ratio: float = 0.2,
                 preprocess: Callable[[str], str] = None,
                 device: str = "cpu",
                 verbose: bool = False):
        self.tokenize = tokenize
        self.topk = topk
        self.min_freq = min_freq
        self.vocab_topk = vocab_topk
        self.max_len = max_len
        self.valid_ratio = valid_ratio
        self.preprocess = preprocess
        self.device = device
        self.verbose = verbose
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.wtoi = dict()
        self.vocab = [self.mask_token,
                      self.pad_token,
                      self.unk_token,
                      self.bos_token,
                      self.eos_token]
        self.special_char_offset = len(self.vocab)

        self.data = {'train': [], 'valid': []}
        self.unigram_probs = None

    @property
    def mask_idx(self):
        return self.wtoi[self.mask_token]

    @property
    def pad_idx(self):
        return self.wtoi[self.pad_token]

    def __process_lines(self, raw_lines: List[str]) -> List[List[str]]:
        tok_pre_sentences = []
        for line in raw_lines:
            if self.preprocess:
                pre_sentence = self.preprocess(line.strip())
            else:
                pre_sentence = line.strip()
            tokenized = self.tokenize(pre_sentence)
            if len(tokenized) <= self.max_len and len(tokenized) > 1:
                tok_pre_sentences.append(self.tokenize(pre_sentence))
        return tok_pre_sentences

    def extract_from_archive(self, corpus_tar_gz: str):
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

        sentences = {'train': [], 'valid': []}
        valid_selector = np.zeros(len(processed_book_sentences))
        num_valid_sentences = int(len(processed_book_sentences) * self.valid_ratio)
        valid_selector[np.random.randint(0, len(processed_book_sentences), num_valid_sentences)] = 1
        sentences['valid'] = list(it.compress(processed_book_sentences, valid_selector))
        sentences['train'] = list(it.compress(processed_book_sentences, 1-valid_selector))
        processed_book_sentences = None

        vocab_count = Counter(it.chain(*sentences['train']))
        if self.vocab_topk == 0:
            n = len(vocab_count)
        else:
            n = self.vocab_topk
        temp_vocab = [w for w, _ in it.takewhile(lambda x: x[1] > self.min_freq, vocab_count.most_common(n))]
        self.vocab.extend(temp_vocab)
        self.wtoi = dict([(w, i) for i, w in enumerate(self.vocab)])

        self.unigram_probs = torch.zeros(len(self.vocab))
        for i, w in enumerate(temp_vocab):
            self.unigram_probs[i + self.special_char_offset] = vocab_count[w]
        self.unigram_probs /= sum([c for _, c in vocab_count.items()])
        self.unigram_probs[self.wtoi[self.unk_token]] = 1. - self.unigram_probs.sum().item()

        vocab_set = set(self.vocab)

        def fill_data(which):
            while len(sentences[which]) > 0:
                s = sentences[which].pop(-1)
                converted = [self.wtoi[self.bos_token]]
                converted.extend([self.wtoi[w] if w in vocab_set else self.wtoi[self.unk_token] for w in s])
                converted.extend([self.wtoi[self.eos_token]])
                self.data[which].append(torch.tensor(converted, dtype=torch.long))
            # for s in sentences[which]:
            #     converted = [self.wtoi[self.bos_token]]
            #     converted.extend([self.wtoi[w] if w in vocab_set else self.wtoi[self.unk_token] for w in s])
            #     converted.extend([self.wtoi[self.eos_token]])
            #     self.data[which].append(torch.tensor(converted, dtype=torch.long))

        print('Converting train data...', end='')
        fill_data('train')
        print('DONE')
        print('Converting validation data...', end='')
        fill_data('valid')
        print('DONE')

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

    def __call__(self,
                 bs: int = 32,
                 which: str = 'train'):
        batch = []
        longest = 0
        for data_counter, data_index in enumerate(torch.randperm(len(self.data[which]))):
            example = self.data[which][data_index]
            if len(example) > longest:
                longest = len(example)
            batch.append(example)
            if (data_counter + 1) % bs == 0 or (data_counter + 1) == len(self.data[which]):
                combined_batch = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.wtoi[self.pad_token]).to(self.device)
                example_lengths = torch.zeros([len(batch)], dtype=torch.long)
                for bi, ex in enumerate(batch):
                    example_lengths[bi] = len(ex)
                    combined_batch[bi, :len(ex)] = ex
                yield combined_batch, example_lengths
                longest = 0
                batch.clear()

    def decode_tensor(self, t: torch.Tensor) -> List[str]:
        if t.dim() == 1:
            t = t.unsqueeze(0)

        decoded_sent = []

        for i in range(t.shape[0]):
            sent = [self.vocab[w.item()] for w in t[i] if self.vocab[w.item()] != self.pad_idx]
            decoded_sent.append(' '.join(sent))

        return decoded_sent



class PretrainingDataset(object):

    def __init__(self,
                 src_ds: CorpusLoader,
                 masking_prob: float = 0.8,
                 random_prob: float = 0.1,
                 keeping_prob: float = 0.1,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.noising_probs = torch.tensor([masking_prob, random_prob, keeping_prob]).to(device)
        assert self.noising_probs.sum() == 1.
        self.device = device

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, lengths in self.src_ds(bs=bs, which=which):
            noised_batch = batch.clone()
            clean_segments = []
            offsets_starts = []
            longest = 0
            for bi in range(batch.shape[0]):
                mask_len = (lengths[bi] - 2) // 2
                if mask_len > longest:
                    longest = mask_len
                mask_start = torch.randint(1, mask_len + 1, [1]).item()
                offsets_starts.append(mask_start)
                clean_segments.append(batch[bi, mask_start:mask_start+mask_len])
                actions = self.noising_probs.multinomial(mask_len, replacement=True)
                for ai, si in enumerate(range(mask_start, mask_start+mask_len)):
                    if actions[ai] == 0:
                        noised_batch[bi, si] = self.src_ds.wtoi[self.src_ds.mask_token]
                    elif actions[ai] == 1:
                        noised_batch[bi, si] = self.src_ds.unigram_probs.multinomial(1).item()

            segments = torch.empty([batch.shape[0], longest], dtype=torch.long).fill_(self.src_ds.wtoi[self.src_ds.pad_token]).to(self.device)
            shifted_segments = torch.empty_like(segments).fill_(self.src_ds.wtoi[self.src_ds.mask_token]).to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            output_key_mask = torch.zeros_like(segments).bool().to(self.device)
            offsets = torch.zeros([batch.shape[0], longest], dtype=torch.long).to(self.device)
            for bi, seg in enumerate(clean_segments):
                segments[bi, :len(seg)] = seg
                shifted_segments[bi, 1:len(seg)] = seg[:-1]
                offsets[bi, :len(seg)] = torch.arange(offsets_starts[bi], offsets_starts[bi] + len(seg))
                input_key_mask[bi, lengths[bi]:] = True
                output_key_mask[bi, len(seg):] = True

            yield noised_batch, input_key_mask, segments, shifted_segments, output_key_mask, offsets


class DirectNoiseDataset(object):

    def __init__(self,
                 src_ds: CorpusLoader,
                 mask_prob: float = 0.5,
                 del_prob: float = 0.15,
                 ins_prob: float = 0.15,
                 keep_prob: float = 0.2,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.action_probs = torch.tensor([mask_prob, del_prob, ins_prob, keep_prob]).to(device)
        assert self.action_probs.sum().allclose(torch.tensor(1.))
        self.device = device

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, lengths in self.src_ds(bs=bs, which=which):
            noised_examples = []
            longest = 0
            for bi, example in enumerate(batch):
                current_noised_example = []
                actions = self.action_probs.multinomial(lengths[bi] - 2, replacement=True)
                current_noised_example.append(example[0])
                for ei, a in enumerate(actions):
                    if a == 0: #mask
                        current_noised_example.append(self.src_ds.mask_idx)
                    elif a == 1: #del
                        continue
                    elif a == 2: #insert
                        current_noised_example.append(example[ei+1])
                        sampled_word = self.src_ds.unigram_probs.multinomial(1).item()
                        current_noised_example.append(sampled_word)
                    elif a == 3: #keep
                        current_noised_example.append(example[ei+1])
                current_noised_example.append(example[lengths[bi]-1])
                if len(current_noised_example) > longest:
                    longest = len(current_noised_example)
                noised_examples.append(torch.tensor(current_noised_example, dtype=torch.long))

            noised_batch = torch.empty([batch.shape[0], longest]).fill_(self.src_ds.pad_idx).to(self.device)
            output_key_mask = torch.zeros_like(batch).bool().to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            for bi, ne in enumerate(noised_examples):
                noised_batch[bi, :ne.shape[0]] = ne
                output_key_mask[bi, lengths[bi]:] = True
                input_key_mask[bi, ne.shape[0]:] = True

            yield noised_batch, input_key_mask, batch, output_key_mask
