import tarfile
import io
import h5py
import torch
import torch.nn as nn
import numpy as np
import itertools as it
from collections import Counter
from typing import Callable, List, Tuple, Dict
import stylecorrection.utils.cython_utils as cu


class H5CorpusLoader(object):

    unk_token = "<unk>"
    mask_token = "<mask>"
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    @classmethod
    def __process_lines(cls,
                        raw_lines: List[str],
                        tokenize: Callable[[str], List[str]],
                        preprocess: Callable[[str], str]) -> List[List[str]]:
        for line in raw_lines:
            if preprocess:
                pre_sentence = preprocess(line.strip())
            else:
                pre_sentence = line.strip()
            tokenized = tokenize(pre_sentence)
            yield tokenized


    @classmethod
    def create_from_compressed(cls,
                               h5_fn: str,
                               corpus_tar_gz: str,
                               tokenize: Callable[[str], List[str]],
                               preprocess: Callable[[str], str] = None,
                               topk: int = 0,
                               max_len: int = 175):
        vocab = [cls.mask_token,
                 cls.pad_token,
                 cls.unk_token,
                 cls.bos_token,
                 cls.eos_token]
        wtoi = dict([(w, i) for i, w in enumerate(vocab)])
        with tarfile.open(corpus_tar_gz, 'r:gz', ignore_zeros=True) as tar_file:
            books = tar_file.getmembers()
            if topk > 0:
                books = books[:topk]
            print('Counting...')
            word_count = 0
            sent_count = 0
            for i, b in enumerate(books):
                print('Processing {}/{} : {}'.format(i, len(books), b.name))
                reader = io.TextIOWrapper(tar_file.extractfile(b))
                raw_text = reader.read(None).splitlines()
                for l in cls.__process_lines(raw_text, tokenize, preprocess):
                    if len(l) > max_len:
                        continue
                    for w in l:
                        if w not in wtoi:
                            wtoi[w] = len(vocab)
                            vocab.append(w)
                    word_count += len(l)
                    sent_count += 1
                print('Sentences count : {}'.format(sent_count))
            print('DONE')

            dt = h5py.string_dtype(encoding='utf-8')
            with h5py.File(h5_fn, 'w') as h5_file:
                corpus_ds = h5_file.create_dataset('corpus', (word_count, ), dtype='u4')
                start_len_ds = h5_file.create_dataset('sentences', (sent_count, 2), dtype='u4')
                vocab_ds = h5_file.create_dataset('vocab', (len(vocab),), dtype=dt)
                vocab_ds[:] = vocab

                position = 0
                line = 0
                for i, b in enumerate(books):
                    print('{:.2%}'.format(float(line)/sent_count))
                    reader = io.TextIOWrapper(tar_file.extractfile(b))
                    raw_text = reader.read(None).splitlines()
                    for l in cls.__process_lines(raw_text, tokenize, preprocess):
                        if len(l) > max_len:
                            continue
                        converted_line = [wtoi[w] for w in l]
                        corpus_ds[position:position+len(l)] = converted_line
                        start_len_ds[line] = (position, position+len(l))
                        position += len(l)
                        line += 1

    @classmethod
    def generate_split(cls,
                       h5_fn: str,
                       valid_ratio: float = 0.2):
        with h5py.File(h5_fn, 'r+') as h5_file:
            valid_selector = np.zeros(h5_file['sentences'].shape[0])
            if valid_ratio < 1.0:
                num_valid_sentences = int(h5_file['sentences'].shape[0] * valid_ratio)
                valid_selector[np.random.randint(0, h5_file['sentences'].shape[0], num_valid_sentences)] = 1
            else:
                valid_selector[int(valid_ratio):] = 1

            if 'splits' not in h5_file:
                dt = h5py.vlen_dtype(np.dtype('int32'))
                splits_ds = h5_file.create_dataset('splits', (20,), dtype=dt)
                splits_ds.attrs['num_splits'] = 0
            else:
                splits_ds = h5_file['splits']

            splits_ds[splits_ds.attrs['num_splits']] = valid_selector
            splits_ds.attrs['num_splits'] += 1

    @classmethod
    def load_and_split(cls,
                       h5_fn: str,
                       use_split_id: int = 0,
                       vocab_topk: int = 0,
                       min_freq: int = 2,
                       forced_vocab: List[str] = None,
                       smoothing_alpha: float = 0.,
                       device: str = "cpu"):
        with h5py.File(h5_fn, 'r') as h5_file:
            print('Loading in memory...', end='')
            corpus = torch.from_numpy(h5_file['corpus'][:].astype(np.int32))
            sentences = torch.from_numpy(h5_file['sentences'][:, :].astype(np.int))
            global_vocab = h5_file['vocab'][:]
            splits_ds = h5_file['splits']
            valid_selector = splits_ds[use_split_id]

            splits = dict()
            splits['valid'] = sentences[valid_selector.nonzero()[0]]
            splits['train'] = sentences[(1-valid_selector).nonzero()[0]]
            print('DONE')

            print('Counting words...', end='')
            vc = np.zeros(len(global_vocab), dtype=np.int32)
            cu.count_words(corpus.numpy(), splits['train'].numpy(), vc)
            vc = torch.from_numpy(vc)
            vocab_counter = Counter(dict([(i, vc[i].item()) for i in range(5, vc.shape[0])]))
            total_count = vc.sum()
            del vc
            if vocab_topk == 0:
                n = len(vocab_counter)
            else:
                n = vocab_topk
            top_vocab_count = [(wi, c) for wi, c in it.takewhile(lambda x: x[1] > min_freq, vocab_counter.most_common(n))]
            vocab_counter = Counter(dict(top_vocab_count))
            print('DONE')

            print('Building vocabulary and mappings...', end='')
            if forced_vocab is not None:
                global_wtoi = dict([(w, i) for i, w in enumerate(global_vocab)])
                temp_vocab = forced_vocab
                wtoi = dict([(w, i) for i, w in enumerate(temp_vocab)])
                gtr_mapping = torch.empty(len(global_vocab), dtype=torch.int).fill_(wtoi[cls.unk_token])
                diff_counter = 0
                for i, w in enumerate(temp_vocab[5:]):
                    if w in global_wtoi:
                        gtr_mapping[global_wtoi[w]] = i + 5
                    else:
                        diff_counter += 1
                print('Forced vocab : {}/{} words not present'.format(diff_counter, len(temp_vocab) - 5))
                # gtr_mapping = dict([(global_wtoi[w], i + 5) for i, w in enumerate(temp_vocab[5:]) if w in global_wtoi])
                rtg_mapping = dict([(i + 5, global_wtoi[w]) for i, w in enumerate(temp_vocab[5:]) if w in global_wtoi])
            else:
                temp_vocab = list(global_vocab[:5])
                temp_vocab.extend([global_vocab[w] for w, _ in top_vocab_count])
                wtoi = dict([(w, i) for i, w in enumerate(temp_vocab)])
                gtr_mapping = torch.empty(len(global_vocab), dtype=torch.int).fill_(wtoi[cls.unk_token])
                for i, (wi, _) in enumerate(top_vocab_count):
                    gtr_mapping[wi] = i + 5
                rtg_mapping = dict([(i + 5, wi) for i, (wi, _) in enumerate(top_vocab_count)])

            print('DONE')

            print('Adjusting corpus for new vocabulary...', end='')
            cu.map_corpus(corpus.numpy(), gtr_mapping.numpy())
            print('DONE')

            print('Computing unigram probabilities...', end='')
            unigram_probs = torch.zeros(len(temp_vocab))
            smoothing = torch.empty(len(temp_vocab)).fill_(smoothing_alpha)
            smoothing[:5] = 0
            for i in range(5, len(temp_vocab)):
                if i in rtg_mapping:
                    unigram_probs[i] = vocab_counter[rtg_mapping[i]]
            unigram_probs = (unigram_probs + smoothing) / (total_count + smoothing.sum())
            unigram_probs[wtoi[cls.unk_token]] = 1. - unigram_probs.sum().item()
            print('DONE')

            return cls(corpus,
                       splits,
                       temp_vocab,
                       wtoi,
                       unigram_probs,
                       device)

    def __init__(self,
                 corpus: torch.Tensor,
                 sentences: Dict[str, torch.Tensor],
                 vocab: List[str],
                 wtoi: Dict[str, int],
                 unigram_probs: torch.Tensor,
                 device: str = "cpu"):
        self.corpus = corpus
        self.sentences = sentences
        self.vocab = vocab
        self.wtoi = wtoi
        self.unigram_probs = unigram_probs
        self.device = device
        self.rand_generator = torch.Generator()

    def set_manual_rnd_seed(self, seed: int):
        self.rand_generator.manual_seed(seed & ((1<<63)-1))

    def dump_vocab(self, vocab_fn: str):
        with h5py.File(vocab_fn, 'w') as h5_file:
            dt = h5py.string_dtype(encoding='utf-8')
            vocab_ds = h5_file.create_dataset('vocab', (len(self.vocab),), dtype=dt)
            prob_ds = h5_file.create_dataset('unigram_prob', (len(self.vocab),), dtype=np.float)
            vocab_ds[:] = self.vocab
            prob_ds[:] = self.unigram_probs

    def __call__(self,
                 bs: int = 32,
                 which: str = 'train'):
        batch = []
        longest = 0
        for data_counter, data_index in enumerate(torch.randperm(self.sentences[which].shape[0], generator=self.rand_generator)):
            s_start, s_end = self.sentences[which][data_index]
            ex_len = s_end - s_start
            example = torch.zeros(ex_len + 2, dtype=torch.long).to(self.device)
            example[0] = self.wtoi[self.bos_token]
            example[-1] = self.wtoi[self.eos_token]
            example[1:-1] = self.corpus[s_start:s_end]
            if len(example) > longest:
                longest = len(example)
            batch.append(example)
            if (data_counter + 1) % bs == 0 or (data_counter + 1) == self.sentences[which].shape[0]:
                yield batch, longest
                longest = 0
                batch.clear()

    def get_num_sentences(self, which):
        return self.sentences[which].shape[0]

    @property
    def mask_idx(self):
        return self.wtoi[self.mask_token]

    @property
    def pad_idx(self):
        return self.wtoi[self.pad_token]

    @property
    def bos_idx(self):
        return self.wtoi[self.bos_token]

    @property
    def eos_idx(self):
        return self.wtoi[self.eos_token]

    def encode_sentence(self,
                        sentence: str):
        encoded = [self.bos_idx]
        for w in sentence.split(' '):
            if w in self.wtoi:
                encoded.append(self.wtoi[w])
            else:
                encoded.append(self.wtoi[self.unk_token])
        encoded.append(self.eos_idx)
        return torch.tensor(encoded, dtype=torch.long, requires_grad=False)

    def decode_tensor(self, t: torch.Tensor) -> List[str]:
        if t.dim() == 1:
            t = t.unsqueeze(0)

        decoded_sent = []

        for i in range(t.shape[0]):
            sent = [self.vocab[w.item()] for w in t[i] if self.vocab[w.item()] != self.pad_idx]
            decoded_sent.append(' '.join(sent))

        return decoded_sent

class StreamingH5CorpusLoader(object):

    unk_token = "<unk>"
    mask_token = "<mask>"
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    @classmethod
    def __process_lines(cls,
                        raw_lines: List[str],
                        tokenize: Callable[[str], List[str]],
                        preprocess: Callable[[str], str]) -> List[List[str]]:
        for line in raw_lines:
            if preprocess:
                pre_sentence = preprocess(line.strip())
            else:
                pre_sentence = line.strip()
            tokenized = tokenize(pre_sentence)
            yield tokenized


    @classmethod
    def create_from_compressed(cls,
                               h5_fn: str,
                               corpus_tar_gz: str,
                               tokenize: Callable[[str], List[str]],
                               preprocess: Callable[[str], str] = None,
                               topk: int = 0,
                               max_len: int = 175):
        vocab = [cls.mask_token,
                 cls.pad_token,
                 cls.unk_token,
                 cls.bos_token,
                 cls.eos_token]
        wtoi = dict([(w, i) for i, w in enumerate(vocab)])
        with tarfile.open(corpus_tar_gz, 'r:gz', ignore_zeros=True) as tar_file:
            books = tar_file.getmembers()
            if topk > 0:
                books = books[:topk]
            print('Counting...')
            word_count = 0
            sent_count = 0
            for i, b in enumerate(books):
                print('Processing {}/{} : {}'.format(i, len(books), b.name))
                reader = io.TextIOWrapper(tar_file.extractfile(b))
                raw_text = reader.read(None).splitlines()
                for l in cls.__process_lines(raw_text, tokenize, preprocess):
                    if len(l) > max_len:
                        continue
                    for w in l:
                        if w not in wtoi:
                            wtoi[w] = len(vocab)
                            vocab.append(w)
                    word_count += len(l)
                    sent_count += 1
                print('Sentences count : {}'.format(sent_count))
            print('DONE')

            dt = h5py.string_dtype(encoding='utf-8')
            with h5py.File(h5_fn, 'w') as h5_file:
                corpus_ds = h5_file.create_dataset('corpus', (word_count, ), dtype='u4')
                start_len_ds = h5_file.create_dataset('sentences', (sent_count, 2), dtype='u4')
                vocab_ds = h5_file.create_dataset('vocab', (len(vocab),), dtype=dt)
                vocab_ds[:] = vocab

                position = 0
                line = 0
                for i, b in enumerate(books):
                    print('{:.2%}'.format(float(line)/sent_count))
                    reader = io.TextIOWrapper(tar_file.extractfile(b))
                    raw_text = reader.read(None).splitlines()
                    for l in cls.__process_lines(raw_text, tokenize, preprocess):
                        if len(l) > max_len:
                            continue
                        converted_line = [wtoi[w] for w in l]
                        corpus_ds[position:position+len(l)] = converted_line
                        start_len_ds[line] = (position, position+len(l))
                        position += len(l)
                        line += 1

    @classmethod
    def generate_split(cls,
                       h5_fn: str,
                       valid_ratio: float = 0.2):
        with h5py.File(h5_fn, 'r+') as h5_file:
            valid_selector = np.zeros(h5_file['sentences'].shape[0])
            if valid_ratio < 1.0:
                num_valid_sentences = int(h5_file['sentences'].shape[0] * valid_ratio)
                valid_selector[np.random.randint(0, h5_file['sentences'].shape[0], num_valid_sentences)] = 1
            else:
                valid_selector[int(valid_ratio):] = 1

            if 'splits' not in h5_file:
                dt = h5py.vlen_dtype(np.dtype('int32'))
                splits_ds = h5_file.create_dataset('splits', (20,), dtype=dt)
                splits_ds.attrs['num_splits'] = 0
            else:
                splits_ds = h5_file['splits']

            splits_ds[splits_ds.attrs['num_splits']] = valid_selector
            splits_ds.attrs['num_splits'] += 1

    @classmethod
    def load_and_split(cls,
                       h5_fn: str,
                       use_split_id: int = 0,
                       vocab_topk: int = 0,
                       min_freq: int = 2,
                       forced_vocab: List[str] = None,
                       smoothing_alpha: float = 0.,
                       device: str = "cpu"):
        with h5py.File(h5_fn, 'r') as h5_file:
            print('Loading in memory...', end='')
            corpus = torch.from_numpy(h5_file['corpus'][:].astype(np.int32))
            sentences = torch.from_numpy(h5_file['sentences'][:, :].astype(np.int))
            global_vocab = h5_file['vocab'][:]
            splits_ds = h5_file['splits']
            valid_selector = splits_ds[use_split_id]

            splits = dict()
            splits['valid'] = sentences[valid_selector.nonzero()[0]]
            splits['train'] = sentences[(1-valid_selector).nonzero()[0]]
            print('DONE')

            print('Counting words...', end='')
            vc = np.zeros(len(global_vocab), dtype=np.int32)
            cu.count_words(corpus.numpy(), splits['train'].numpy(), vc)
            vc = torch.from_numpy(vc)
            vocab_counter = Counter(dict([(i, vc[i].item()) for i in range(5, vc.shape[0])]))
            total_count = vc.sum()
            del vc
            if vocab_topk == 0:
                n = len(vocab_counter)
            else:
                n = vocab_topk
            top_vocab_count = [(wi, c) for wi, c in it.takewhile(lambda x: x[1] > min_freq, vocab_counter.most_common(n))]
            vocab_counter = Counter(dict(top_vocab_count))
            print('DONE')

            print('Building vocabulary and mappings...', end='')
            if forced_vocab is not None:
                global_wtoi = dict([(w, i) for i, w in enumerate(global_vocab)])
                temp_vocab = forced_vocab
                wtoi = dict([(w, i) for i, w in enumerate(temp_vocab)])
                gtr_mapping = torch.empty(len(global_vocab), dtype=torch.int).fill_(wtoi[cls.unk_token])
                diff_counter = 0
                for i, w in enumerate(temp_vocab[5:]):
                    if w in global_wtoi:
                        gtr_mapping[global_wtoi[w]] = i + 5
                    else:
                        diff_counter += 1
                print('Forced vocab : {}/{} words not present'.format(diff_counter, len(temp_vocab) - 5))
                # gtr_mapping = dict([(global_wtoi[w], i + 5) for i, w in enumerate(temp_vocab[5:]) if w in global_wtoi])
                rtg_mapping = dict([(i + 5, global_wtoi[w]) for i, w in enumerate(temp_vocab[5:]) if w in global_wtoi])
            else:
                temp_vocab = list(global_vocab[:5])
                temp_vocab.extend([global_vocab[w] for w, _ in top_vocab_count])
                wtoi = dict([(w, i) for i, w in enumerate(temp_vocab)])
                gtr_mapping = torch.empty(len(global_vocab), dtype=torch.int).fill_(wtoi[cls.unk_token])
                for i, (wi, _) in enumerate(top_vocab_count):
                    gtr_mapping[wi] = i + 5
                rtg_mapping = dict([(i + 5, wi) for i, (wi, _) in enumerate(top_vocab_count)])

            print('DONE')

            print('Adjusting corpus for new vocabulary...', end='')
            cu.map_corpus(corpus.numpy(), gtr_mapping.numpy())
            print('DONE')

            print('Computing unigram probabilities...', end='')
            unigram_probs = torch.zeros(len(temp_vocab)).to(device)
            smoothing = torch.empty(len(temp_vocab)).fill_(smoothing_alpha).to(device)
            smoothing[:5] = 0
            for i in range(5, len(temp_vocab)):
                if i in rtg_mapping:
                    unigram_probs[i] = vocab_counter[rtg_mapping[i]]
            unigram_probs = (unigram_probs + smoothing) / (total_count + smoothing.sum())
            unigram_probs[wtoi[cls.unk_token]] = 1. - unigram_probs.sum().item()
            print('DONE')

            train_ds = cls(corpus,
                           splits['train'],
                           temp_vocab,
                           wtoi,
                           unigram_probs,
                           device)

            valid_ds = cls(corpus,
                           splits['valid'],
                           temp_vocab,
                           wtoi,
                           unigram_probs,
                           device)

            return train_ds, valid_ds

    def __init__(self,
                 corpus: torch.Tensor,
                 sentences: torch.Tensor,
                 vocab: List[str],
                 wtoi: Dict[str, int],
                 unigram_probs: torch.Tensor,
                 device: str = "cpu"):
        self.corpus = corpus
        self.sentences = sentences
        self.vocab = vocab
        self.wtoi = wtoi
        self.unigram_probs = unigram_probs
        self.device = device
        self.current_iterating_order = None
        self.current_iterating_idx = 0
        self.generate_iterating_order = True

    def dump_vocab(self, vocab_fn: str):
        with h5py.File(vocab_fn, 'w') as h5_file:
            dt = h5py.string_dtype(encoding='utf-8')
            vocab_ds = h5_file.create_dataset('vocab', (len(self.vocab),), dtype=dt)
            prob_ds = h5_file.create_dataset('unigram_prob', (len(self.vocab),), dtype=np.float)
            vocab_ds[:] = self.vocab
            prob_ds[:] = self.unigram_probs

    def __len__(self):
        return self.sentences.shape[0]

    def __iter__(self):
        if self.generate_iterating_order:
            self.current_iterating_idx = 0
            self.current_iterating_order = torch.randperm(self.sentences.shape[0])
        for self.current_iterating_idx in range(self.current_iterating_idx, len(self)):
            s_start, s_end = self.sentences[self.current_iterating_order[self.current_iterating_idx]]
            ex_len = s_end - s_start
            example = torch.zeros(ex_len + 2, dtype=torch.long).to(self.device)
            example[0] = self.wtoi[self.bos_token]
            example[-1] = self.wtoi[self.eos_token]
            example[1:-1] = self.corpus[s_start:s_end]
            yield example

        self.generate_iterating_order = True

    @property
    def mask_idx(self):
        return self.wtoi[self.mask_token]

    @property
    def pad_idx(self):
        return self.wtoi[self.pad_token]

    @property
    def bos_idx(self):
        return self.wtoi[self.bos_token]

    @property
    def eos_idx(self):
        return self.wtoi[self.eos_token]

    def encode_sentence(self,
                        sentence: str):
        encoded = [self.bos_idx]
        for w in sentence.split(' '):
            if w in self.wtoi:
                encoded.append(self.wtoi[w])
            else:
                encoded.append(self.wtoi[self.unk_token])
        encoded.append(self.eos_idx)
        return torch.tensor(encoded, dtype=torch.long, requires_grad=False)

    def decode_tensor(self, t: torch.Tensor) -> List[str]:
        if t.dim() == 1:
            t = t.unsqueeze(0)

        decoded_sent = []

        for i in range(t.shape[0]):
            sent = []
            for w in t[i]:
                vocab_token = self.vocab[w.item()]
                sent.append(vocab_token)
                if vocab_token == self.eos_token:
                    break
            decoded_sent.append(' '.join(sent))

        return decoded_sent


class PretrainingDataset(object):

    def __init__(self,
                 src_ds: H5CorpusLoader,
                 masking_prob: float = 0.8,
                 random_prob: float = 0.1,
                 keeping_prob: float = 0.1,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.noising_probs = torch.tensor([masking_prob, random_prob, keeping_prob]).to(device)
        assert self.noising_probs.sum() == 1.
        self.device = device

    def get_num_sentences(self, which):
        return self.src_ds.get_num_sentences(which)

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, longest_clean in self.src_ds(bs=bs, which=which):
            noised_batch = nn.utils.rnn.pad_sequence(batch, True, self.src_ds.pad_idx).to(self.device)
            clean_segments = []
            offsets_starts = []
            longest = 0
            for bi, example in enumerate(batch):
                mask_len = (example.shape[0] - 2) // 2
                if mask_len > longest:
                    longest = mask_len
                if mask_len > 0:
                    mask_start = torch.randint(1, mask_len + 1, [1]).item()
                else:
                    mask_start = 1
                    mask_len = 1
                offsets_starts.append(mask_start)
                clean_segments.append(example[mask_start:mask_start+mask_len])
                actions = self.noising_probs.multinomial(mask_len, replacement=True)
                for ai, si in enumerate(range(mask_start, mask_start+mask_len)):
                    if actions[ai] == 0:
                        noised_batch[bi, si] = self.src_ds.wtoi[self.src_ds.mask_token]
                    elif actions[ai] == 1:
                        noised_batch[bi, si] = self.src_ds.unigram_probs.multinomial(1).item()

            if longest == 0:
                print('longest 0?!')
                continue
            segments = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.src_ds.wtoi[self.src_ds.pad_token]).to(self.device)
            shifted_segments = torch.empty_like(segments).fill_(self.src_ds.wtoi[self.src_ds.mask_token]).to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            output_key_mask = torch.zeros_like(segments).bool().to(self.device)
            offsets = torch.zeros([len(batch), longest], dtype=torch.long).to(self.device)
            for bi, seg in enumerate(clean_segments):
                segments[bi, :len(seg)] = seg
                shifted_segments[bi, 1:len(seg)] = seg[:-1]
                offsets[bi, :len(seg)] = torch.arange(offsets_starts[bi], offsets_starts[bi] + len(seg))
                input_key_mask[bi, batch[bi].shape[0]:] = True
                output_key_mask[bi, len(seg):] = True

            yield noised_batch, input_key_mask, segments, shifted_segments, output_key_mask, offsets

class MASSPretrainingDataset(object):

    def __init__(self,
                 src_ds: StreamingH5CorpusLoader,
                 masking_prob: float = 0.8,
                 random_prob: float = 0.1,
                 keeping_prob: float = 0.1,
                 tokens_per_batch: int = 1000,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.noising_probs = torch.tensor([masking_prob, random_prob, keeping_prob]).to(device)
        assert self.noising_probs.sum() == 1.
        self.tokens_per_batch = tokens_per_batch
        self.device = device

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, longest_clean in self.src_ds:
            noised_batch = nn.utils.rnn.pad_sequence(batch, True, self.src_ds.pad_idx).to(self.device)
            clean_segments = []
            offsets_starts = []
            longest = 0
            for bi, example in enumerate(batch):
                mask_len = (example.shape[0] - 2) // 2
                if mask_len > longest:
                    longest = mask_len
                if mask_len > 0:
                    mask_start = torch.randint(1, mask_len + 1, [1]).item()
                else:
                    mask_start = 1
                    mask_len = 1
                offsets_starts.append(mask_start)
                clean_segments.append(example[mask_start:mask_start+mask_len])
                actions = self.noising_probs.multinomial(mask_len, replacement=True)
                for ai, si in enumerate(range(mask_start, mask_start+mask_len)):
                    if actions[ai] == 0:
                        noised_batch[bi, si] = self.src_ds.wtoi[self.src_ds.mask_token]
                    elif actions[ai] == 1:
                        noised_batch[bi, si] = self.src_ds.unigram_probs.multinomial(1).item()

            if longest == 0:
                print('longest 0?!')
                continue
            segments = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.src_ds.wtoi[self.src_ds.pad_token]).to(self.device)
            shifted_segments = torch.empty_like(segments).fill_(self.src_ds.wtoi[self.src_ds.mask_token]).to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            output_key_mask = torch.zeros_like(segments).bool().to(self.device)
            offsets = torch.zeros([len(batch), longest], dtype=torch.long).to(self.device)
            for bi, seg in enumerate(clean_segments):
                segments[bi, :len(seg)] = seg
                shifted_segments[bi, 1:len(seg)] = seg[:-1]
                offsets[bi, :len(seg)] = torch.arange(offsets_starts[bi], offsets_starts[bi] + len(seg))
                input_key_mask[bi, batch[bi].shape[0]:] = True
                output_key_mask[bi, len(seg):] = True

            yield noised_batch, input_key_mask, segments, shifted_segments, output_key_mask, offsets

class BARTPretrainingDataset(object):

    def __init__(self,
                 src_ds: H5CorpusLoader,
                 masking_ratio: float = 0.3,
                 poisson_lambda: float = 3,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.masking_ratio = masking_ratio
        self.poisson_dist = torch.distributions.Poisson(poisson_lambda)
        self.device = device

    def get_num_sentences(self, which):
        return self.src_ds.get_num_sentences(which)

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, longest_clean in self.src_ds(bs=bs, which=which):
            clean_segments = []
            offsets_starts = []
            longest_masked = 0
            masked_examples = []
            for bi, example in enumerate(batch):
                num_mask = int((example.shape[0] - 2) * self.masking_ratio)
                samples = self.poisson_dist.sample([example.shape[0]*2]).int()
                cummul = samples.cumsum(0)
                si = (cummul >= num_mask).nonzero().squeeze(1)[0]
                samples[si] -= (cummul[si] - num_mask)
                end_padding = samples[list(range(si, -1, -1))].cumsum(0)
                example_idx = 1
                masked_example_idx = 1
                masked_example = torch.zeros([example.shape[0] - num_mask + si + 1], dtype=torch.long).to(self.device)
                masked_example[[0, -1]] = example[[0, -1]]
                if masked_example.shape[0] > longest_masked:
                    longest_masked = masked_example.shape[0]
                for i in range(si + 1):
                    mask_start = torch.randint(low=example_idx, high=(example.shape[0]-end_padding[si - i]).int().item(), size=[1]).item()
                    copy_len = mask_start - example_idx
                    masked_example[masked_example_idx:masked_example_idx + copy_len] = example[example_idx:mask_start]
                    masked_example[masked_example_idx+copy_len] = self.src_ds.mask_idx
                    masked_example_idx += copy_len + 1
                    example_idx = mask_start + samples[i]
                masked_example[masked_example_idx:-1] = example[example_idx:-1]
                masked_examples.append(masked_example)

            if longest_masked == 0:
                print('longest 0?!')
                continue

            masked_batch = torch.empty([len(batch), longest_masked], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            input_key_mask = torch.zeros_like(masked_batch).bool().to(self.device)
            clean_shifted_bos = torch.empty([len(batch), longest_clean - 1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            output_key_mask = torch.zeros_like(clean_shifted_bos).bool().to(self.device)
            clean_shifted_eos = torch.empty([len(batch), longest_clean - 1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            for bi, me in enumerate(masked_examples):
                masked_batch[bi, :me.shape[0]] = me
                input_key_mask[bi, me.shape[0]:] = True
                clean_shifted_bos[bi, :batch[bi].shape[0]-1] = batch[bi][:-1]
                clean_shifted_eos[bi, :batch[bi].shape[0]-1] = batch[bi][1:]
                output_key_mask[bi, batch[bi].shape[0]-1:] = True

            yield masked_batch, input_key_mask, clean_shifted_eos, clean_shifted_bos, output_key_mask, None


class DirectNoiseDataset(object):

    def __init__(self,
                 src_ds: H5CorpusLoader,
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

        for batch, longest_clean in self.src_ds(bs=bs, which=which):
            noised_examples = []
            longest = 0
            for bi, example in enumerate(batch):
                current_noised_example = []
                actions = self.action_probs.multinomial(example.shape[0] - 2, replacement=True)
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
                current_noised_example.append(example[-1])
                if len(current_noised_example) > longest:
                    longest = len(current_noised_example)
                noised_examples.append(torch.tensor(current_noised_example, dtype=torch.long))

            noised_batch = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            bos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            eos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            output_key_mask = torch.zeros_like(eos_trunc).bool().to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            for bi, ne in enumerate(noised_examples):
                bos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][1:]
                eos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][:-1]
                noised_batch[bi, :ne.shape[0]] = ne
                output_key_mask[bi, batch[bi].shape[0]-1:] = True
                input_key_mask[bi, ne.shape[0]:] = True

            yield noised_batch, input_key_mask, eos_trunc, bos_trunc, output_key_mask

    def get_num_sentences(self, which):
        return self.src_ds.get_num_sentences(which)

class ParallelTextDataset(object):

    def __init__(self,
                 left_ds: H5CorpusLoader,
                 right_ds: H5CorpusLoader,
                 device: str = 'cpu'):
        self.left_ds = left_ds
        self.right_ds = right_ds
        self.device = device
        common_rnd_seed = 42
        self.left_ds.set_manual_rnd_seed(common_rnd_seed)
        self.right_ds.set_manual_rnd_seed(common_rnd_seed)

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for (batch_left, longest_left), (batch_right, longest_right) in zip(self.left_ds(bs=bs, which=which), self.left_ds(bs=bs, which=which)):
            return batch_left, batch_right

class CANoiseDataset(object):

    def __init__(self,
                 src_ds: H5CorpusLoader,
                 replace_prob: float = 0.1,
                 del_prob: float = 0.1,
                 ins_prob: float = 0.1,
                 keep_prob: float = 0.3,
                 mask_prob: float = 0.4,
                 sigma: float = 0.5,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.action_probs = torch.tensor([replace_prob, del_prob, ins_prob, keep_prob, mask_prob]).to(device)
        self.sigma = sigma
        assert self.action_probs.sum().allclose(torch.tensor(1.))
        self.device = device

    def __call__(self,
                 bs: int = 32,
                 which: str = "train"):

        for batch, longest_clean in self.src_ds(bs=bs, which=which):
            noised_examples = []
            longest = 0
            for bi, example in enumerate(batch):
                current_noised_example = []
                actions = self.action_probs.multinomial(example.shape[0] - 2, replacement=True)
                for ei, a in enumerate(actions):
                    if a == 0: #replace
                        current_noised_example.append(self.src_ds.unigram_probs.multinomial(1).item())
                    elif a == 1: #del
                        continue
                    elif a == 2: #insert
                        current_noised_example.append(example[ei+1])
                        sampled_word = self.src_ds.unigram_probs.multinomial(1).item()
                        current_noised_example.append(sampled_word)
                    elif a == 3: #keep
                        current_noised_example.append(example[ei+1])
                    elif a == 4: #mask
                        current_noised_example.append(self.src_ds.mask_idx)
                ne = torch.zeros(len(current_noised_example) + 2, dtype=torch.long)
                current_noised_example = torch.tensor(current_noised_example, dtype=torch.long)
                ne[0] = example[0]
                ne[-1] = example[-1]
                shuffled_indexes = np.array([i + np.random.normal(loc=0, scale=self.sigma) for i in range(len(current_noised_example))]).argsort()
                ne[1:-1] = current_noised_example[shuffled_indexes]
                if ne.shape[0] > longest:
                    longest = ne.shape[0]
                noised_examples.append(ne)

            noised_batch = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            bos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            eos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
            output_key_mask = torch.zeros_like(eos_trunc).bool().to(self.device)
            input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
            for bi, ne in enumerate(noised_examples):
                bos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][1:]
                eos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][:-1]
                noised_batch[bi, :ne.shape[0]] = ne
                output_key_mask[bi, batch[bi].shape[0]-1:] = True
                input_key_mask[bi, ne.shape[0]:] = True

            yield noised_batch, input_key_mask, eos_trunc, bos_trunc, output_key_mask

    def get_num_sentences(self, which):
        return self.src_ds.get_num_sentences(which)

class StreamingBaseDataset(object):

    def __init__(self,
                 src_ds: StreamingH5CorpusLoader,
                 tokens_per_batch: int = 1000,
                 offset_padding: int = 4999,
                 device: str = "cpu"):
        self.src_ds = src_ds
        self.tokens_per_batch = tokens_per_batch
        self.offset_padding = offset_padding
        self.device = device

    def __iter__(self):
        clean_examples_iter = iter(self.src_ds)

        current_batch_dec_in = []
        current_batch_enc_in = []
        current_batch_dec_out = []
        current_batch_offsets = []
        current_longest_dec_in = 0
        current_longest_enc_in = 0
        current_longest_dec_out = 0

        examples_exhausted = False
        while not examples_exhausted:
            try:
                example = next(clean_examples_iter)
                enc_input, dec_input, dec_output, offsets = self.process_example(example)
            except StopIteration:
                examples_exhausted = True
            batch_token_count = max(current_longest_enc_in, enc_input.shape[0]) * (len(current_batch_enc_in) + 1) + max(current_longest_dec_in, dec_input.shape[0]) * (len(current_batch_enc_in) + 1)
            if batch_token_count > self.tokens_per_batch or examples_exhausted:
                enc_in_bacth = torch.empty([len(current_batch_enc_in), current_longest_enc_in], dtype=torch.long).fill_(
                    self.src_ds.pad_idx).to(self.device)
                dec_in_batch = torch.empty([len(current_batch_dec_in), current_longest_dec_in], dtype=torch.long).fill_(
                    self.src_ds.pad_idx).to(self.device)
                dec_out_batch = torch.empty([len(current_batch_dec_out), current_longest_dec_out], dtype=torch.long).fill_(
                    self.src_ds.pad_idx).to(self.device)
                output_key_mask = torch.zeros_like(dec_in_batch).bool().to(self.device)
                input_key_mask = torch.zeros_like(enc_in_bacth).bool().to(self.device)

                if current_batch_offsets[0] is None:
                    offsets_batch = None
                else:
                    offsets_batch = torch.empty([len(dec_in_batch), current_longest_dec_in], dtype=torch.long).fill_(self.offset_padding).to(self.device)

                for bi, ne in enumerate(current_batch_enc_in):
                    dec_in_batch[bi, :current_batch_dec_in[bi].shape[0]] = current_batch_dec_in[bi]
                    dec_out_batch[bi, :current_batch_dec_out[bi].shape[0]] = current_batch_dec_out[bi]
                    enc_in_bacth[bi, :ne.shape[0]] = ne
                    output_key_mask[bi, current_batch_dec_in[bi].shape[0]:] = True
                    input_key_mask[bi, ne.shape[0]:] = True
                    if current_batch_offsets[bi] is not None:
                        offsets_batch[bi, :current_batch_offsets[bi].shape[0]] = current_batch_offsets[bi]

                yield enc_in_bacth, input_key_mask, dec_out_batch, dec_in_batch, output_key_mask, offsets_batch
                current_batch_dec_in.clear()
                current_batch_enc_in.clear()
                current_batch_dec_out.clear()
                current_batch_offsets.clear()
                current_longest_dec_in = 0
                current_longest_enc_in = 0
                current_longest_dec_out = 0

            if enc_input.shape[0] > current_longest_enc_in:
                current_longest_enc_in = enc_input.shape[0]
            if dec_input.shape[0] > current_longest_dec_in:
                current_longest_dec_in = dec_input.shape[0]
            if dec_output.shape[0] > current_longest_dec_out:
                current_longest_dec_out = dec_output.shape[0]

            current_batch_enc_in.append(enc_input)
            current_batch_dec_in.append(dec_input)
            current_batch_dec_out.append(dec_output)
            current_batch_offsets.append(offsets)


    def process_example(self, example: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class StreamingCANoiseDataset(StreamingBaseDataset):

    def __init__(self,
                 src_ds: StreamingH5CorpusLoader,
                 replace_prob: float = 0.1,
                 del_prob: float = 0.1,
                 ins_prob: float = 0.1,
                 keep_prob: float = 0.3,
                 mask_prob: float = 0.4,
                 sigma: float = 0.5,
                 tokens_per_batch: int = 1000,
                 device: str = "cpu"):
        super(StreamingCANoiseDataset, self).__init__(src_ds, tokens_per_batch=tokens_per_batch, device=device)
        self.action_probs = torch.tensor([replace_prob, del_prob, ins_prob, keep_prob, mask_prob]).to(device)
        self.sigma = sigma
        assert self.action_probs.sum().allclose(torch.tensor(1.))

    def process_example(self, example: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_noised_example = []
        actions = self.action_probs.multinomial(example.shape[0] - 2, replacement=True)
        for ei, a in enumerate(actions):
            if a == 0:  # replace
                current_noised_example.append(self.src_ds.unigram_probs.multinomial(1).item())
            elif a == 1:  # del
                continue
            elif a == 2:  # insert
                current_noised_example.append(example[ei + 1])
                sampled_word = self.src_ds.unigram_probs.multinomial(1).item()
                current_noised_example.append(sampled_word)
            elif a == 3:  # keep
                current_noised_example.append(example[ei + 1])
            elif a == 4:  # mask
                current_noised_example.append(self.src_ds.mask_idx)
        ne = torch.zeros(len(current_noised_example) + 2, dtype=torch.long)
        current_noised_example = torch.tensor(current_noised_example, dtype=torch.long)
        ne[[0, -1]] = example[[0, -1]]
        shuffled_indexes = np.array(
            [i + np.random.normal(loc=0, scale=self.sigma) for i in range(len(current_noised_example))]).argsort()
        ne[1:-1] = current_noised_example[shuffled_indexes]
        return ne, example[:-1], example[1:], None

class StreamingMASSPretrainingDataset(StreamingBaseDataset):

    def __init__(self,
                 src_ds: StreamingH5CorpusLoader,
                 masking_prob: float = 0.8,
                 random_prob: float = 0.1,
                 keeping_prob: float = 0.1,
                 tokens_per_batch: int = 1000,
                 device: str = "cpu"):
        super(StreamingMASSPretrainingDataset, self).__init__(src_ds, tokens_per_batch=tokens_per_batch, device=device)
        self.noising_probs = torch.tensor([masking_prob, random_prob, keeping_prob]).to(device)
        assert self.noising_probs.sum() == 1.

    def process_example(self, example: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_len = (example.shape[0] - 2) // 2
        if mask_len > 0:
            mask_start = torch.randint(1, mask_len + 1, [1]).item()
        else:
            mask_start = 1
            mask_len = 1
        masked_example = example.clone()
        masked_slice = example[mask_start:mask_start+mask_len]
        shifted_masked_slice = torch.zeros_like(masked_slice)
        shifted_masked_slice[0] = self.src_ds.mask_idx
        shifted_masked_slice[1:] = masked_slice[:-1]
        actions = self.noising_probs.multinomial(mask_len, replacement=True)
        mask_indices = (actions == 0).nonzero().squeeze(1)
        replace_indices = (actions == 1).nonzero().squeeze(1)
        masked_example[mask_indices+mask_start] = self.src_ds.mask_idx
        if replace_indices.shape[0] > 0:
            masked_example[replace_indices] = self.src_ds.unigram_probs.multinomial(replace_indices.shape[0])
        # for ai, si in enumerate(range(mask_start, mask_start + mask_len)):
        #     if actions[ai] == 0:
        #         masked_example[si] = self.src_ds.wtoi[self.src_ds.mask_token]
        #     elif actions[ai] == 1:
        #         masked_example[si] = self.src_ds.unigram_probs.multinomial(1).item()

        return masked_example, shifted_masked_slice, masked_slice, torch.arange(mask_start, mask_start+mask_len)

class StreamingBARTPretrainingDataset(StreamingBaseDataset):

    def __init__(self,
                 src_ds: StreamingH5CorpusLoader,
                 masking_ratio: float = 0.3,
                 poisson_lambda: float = 3,
                 tokens_per_batch: int = 1000,
                 device: str = "cpu"):
        super(StreamingBARTPretrainingDataset, self).__init__(src_ds, tokens_per_batch=tokens_per_batch, device=device)
        self.masking_ratio = masking_ratio
        self.poisson_dist = torch.distributions.Poisson(poisson_lambda)

    def process_example(self, example: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_mask = int((example.shape[0] - 2) * self.masking_ratio)
        samples = self.poisson_dist.sample([example.shape[0] * 2]).int()
        cummul = samples.cumsum(0)
        si = (cummul >= num_mask).nonzero().squeeze(1)[0]
        samples[si] -= (cummul[si] - num_mask)
        end_padding = samples[list(range(si, -1, -1))].cumsum(0)
        example_idx = 1
        masked_example_idx = 1
        masked_example = torch.zeros([example.shape[0] - num_mask + si + 1], dtype=torch.long).to(self.device)
        masked_example[[0, -1]] = example[[0, -1]]
        for i in range(si + 1):
            mask_start = torch.randint(low=example_idx, high=(example.shape[0] - end_padding[si - i]).int().item(),
                                       size=[1]).item()
            copy_len = mask_start - example_idx
            masked_example[masked_example_idx:masked_example_idx + copy_len] = example[example_idx:mask_start]
            masked_example[masked_example_idx + copy_len] = self.src_ds.mask_idx
            masked_example_idx += copy_len + 1
            example_idx = mask_start + samples[i]
        masked_example[masked_example_idx:-1] = example[example_idx:-1]
        return masked_example, example[:-1], example[1:], None

# class StreamingCANoiseDataset(object):
#
#     def __init__(self,
#                  src_ds: StreamingH5CorpusLoader,
#                  replace_prob: float = 0.1,
#                  del_prob: float = 0.1,
#                  ins_prob: float = 0.1,
#                  keep_prob: float = 0.3,
#                  mask_prob: float = 0.4,
#                  sigma: float = 0.5,
#                  tokens_per_batch: int = 1000,
#                  device: str = "cpu"):
#         self.src_ds = src_ds
#         self.action_probs = torch.tensor([replace_prob, del_prob, ins_prob, keep_prob, mask_prob]).to(device)
#         self.sigma = sigma
#         assert self.action_probs.sum().allclose(torch.tensor(1.))
#         self.tokens_per_batch = tokens_per_batch
#         self.device = device
#
#     def noise_example(self, example: torch.Tensor):
#         current_noised_example = []
#         actions = self.action_probs.multinomial(example.shape[0] - 2, replacement=True)
#         for ei, a in enumerate(actions):
#             if a == 0:  # replace
#                 current_noised_example.append(self.src_ds.unigram_probs.multinomial(1).item())
#             elif a == 1:  # del
#                 continue
#             elif a == 2:  # insert
#                 current_noised_example.append(example[ei + 1])
#                 sampled_word = self.src_ds.unigram_probs.multinomial(1).item()
#                 current_noised_example.append(sampled_word)
#             elif a == 3:  # keep
#                 current_noised_example.append(example[ei + 1])
#             elif a == 4:  # mask
#                 current_noised_example.append(self.src_ds.mask_idx)
#         ne = torch.zeros(len(current_noised_example) + 2, dtype=torch.long)
#         current_noised_example = torch.tensor(current_noised_example, dtype=torch.long)
#         ne[[0, -1]] = example[[0, -1]]
#         shuffled_indexes = np.array(
#             [i + np.random.normal(loc=0, scale=self.sigma) for i in range(len(current_noised_example))]).argsort()
#         ne[1:-1] = current_noised_example[shuffled_indexes]
#         return ne
#
#     def __iter__(self):
#
#         clean_examples_iter = iter(self.src_ds)
#
#         current_batch_clean = []
#         current_batch_noised = []
#         current_longest_clean = 0
#         current_longest_noised = 0
#
#         examples_exhausted = False
#         while not examples_exhausted:
#             try:
#                 example = next(clean_examples_iter)
#                 noised_example = self.noise_example(example)
#             except StopIteration:
#                 examples_exhausted = True
#             batch_token_count = max(current_longest_noised, noised_example.shape[0]) * (len(current_batch_noised) + 1) + max(current_longest_clean, example.shape[0]) * (len(current_batch_clean) + 1)
#             if batch_token_count > self.tokens_per_batch or examples_exhausted:
#                 noised_batch = torch.empty([len(current_batch_noised), current_longest_noised], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
#                 bos_trunc = torch.empty([len(current_batch_clean), current_longest_clean - 1], dtype=torch.long).fill_(
#                     self.src_ds.pad_idx).to(self.device)
#                 eos_trunc = torch.empty([len(current_batch_clean), current_longest_clean - 1], dtype=torch.long).fill_(
#                     self.src_ds.pad_idx).to(self.device)
#                 output_key_mask = torch.zeros_like(eos_trunc).bool().to(self.device)
#                 input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
#                 for bi, ne in enumerate(current_batch_noised):
#                     bos_trunc[bi, :current_batch_clean[bi].shape[0] - 1] = current_batch_clean[bi][1:]
#                     eos_trunc[bi, :current_batch_clean[bi].shape[0] - 1] = current_batch_clean[bi][:-1]
#                     noised_batch[bi, :ne.shape[0]] = ne
#                     output_key_mask[bi, current_batch_clean[bi].shape[0] - 1:] = True
#                     input_key_mask[bi, ne.shape[0]:] = True
#
#                 yield noised_batch, input_key_mask, eos_trunc, bos_trunc, output_key_mask
#                 current_batch_clean.clear()
#                 current_batch_noised.clear()
#                 current_longest_clean = 0
#                 current_longest_noised = 0
#
#             if example.shape[0] > current_longest_clean:
#                 current_longest_clean = example.shape[0]
#             if noised_example.shape[0] > current_longest_noised:
#                 current_longest_noised = noised_example.shape[0]
#             current_batch_clean.append(example)
#             current_batch_noised.append(noised_example)
#
#     def __call__(self,
#                  bs: int = 32,
#                  which: str = "train"):
#
#         for batch, longest_clean in self.src_ds(bs=bs, which=which):
#             noised_examples = []
#             longest = 0
#             for bi, example in enumerate(batch):
#                 current_noised_example = []
#                 actions = self.action_probs.multinomial(example.shape[0] - 2, replacement=True)
#                 for ei, a in enumerate(actions):
#                     if a == 0: #replace
#                         current_noised_example.append(self.src_ds.unigram_probs.multinomial(1).item())
#                     elif a == 1: #del
#                         continue
#                     elif a == 2: #insert
#                         current_noised_example.append(example[ei+1])
#                         sampled_word = self.src_ds.unigram_probs.multinomial(1).item()
#                         current_noised_example.append(sampled_word)
#                     elif a == 3: #keep
#                         current_noised_example.append(example[ei+1])
#                     elif a == 4: #mask
#                         current_noised_example.append(self.src_ds.mask_idx)
#                 ne = torch.zeros(len(current_noised_example) + 2, dtype=torch.long)
#                 current_noised_example = torch.tensor(current_noised_example, dtype=torch.long)
#                 ne[0] = example[0]
#                 ne[-1] = example[-1]
#                 shuffled_indexes = np.array([i + np.random.normal(loc=0, scale=self.sigma) for i in range(len(current_noised_example))]).argsort()
#                 ne[1:-1] = current_noised_example[shuffled_indexes]
#                 if ne.shape[0] > longest:
#                     longest = ne.shape[0]
#                 noised_examples.append(ne)
#
#             noised_batch = torch.empty([len(batch), longest], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
#             bos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
#             eos_trunc = torch.empty([len(batch), longest_clean-1], dtype=torch.long).fill_(self.src_ds.pad_idx).to(self.device)
#             output_key_mask = torch.zeros_like(eos_trunc).bool().to(self.device)
#             input_key_mask = torch.zeros_like(noised_batch).bool().to(self.device)
#             for bi, ne in enumerate(noised_examples):
#                 bos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][1:]
#                 eos_trunc[bi, :batch[bi].shape[0]-1] = batch[bi][:-1]
#                 noised_batch[bi, :ne.shape[0]] = ne
#                 output_key_mask[bi, batch[bi].shape[0]-1:] = True
#                 input_key_mask[bi, ne.shape[0]:] = True
#
#             yield noised_batch, input_key_mask, eos_trunc, bos_trunc, output_key_mask
#
#     def get_num_sentences(self, which):
#         return self.src_ds.get_num_sentences(which)