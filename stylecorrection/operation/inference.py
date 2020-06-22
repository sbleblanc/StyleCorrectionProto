import os
import h5py
from stylecorrection.operation.common import Operation
from stylecorrection.loaders.corpus import *
from stylecorrection.models.transformer import TransformerS2S


class InferenceOperation(Operation):

    @classmethod
    def infer(cls,
              cl: StreamingH5CorpusLoader,
              line: str,
              model: TransformerS2S,
              beam_width: int,
              max_len_scale: int,
              noising_beta: float,
              temperature: float,
              device: str):

        encoded = cl.encode_sentence(line).to(device)

        beam_decoded = model.beam_decode_3(
            encoded,
            torch.tensor([cl.bos_idx], dtype=torch.long).to(device),
            beam_width=beam_width,
            max_len=int(encoded.shape[0] * max_len_scale),
            end_token=cl.eos_idx,
            noising_beta=noising_beta,
            temperature=temperature,
            top_only=False,
            device=device
        )

        return cl.decode_tensor(beam_decoded[0])[0]

    def __init__(self,
                 config: dict,
                 device: str):
        super(InferenceOperation, self).__init__(config, device)
        if config['inference']['force_cpu']:
            self.device = 'cpu'

        if config['inference']['preprocess']:
            from stylecorrection.utils.preprocess import SpacyBPEPreprocess
            codes_fn = os.path.expandvars(config['preprocess']['bpe_codes_fn'])
            bpe_vocab_fn = os.path.expandvars(config['preprocess']['bpe_vocab_fn'])
            self.spacy_bpe_pp = SpacyBPEPreprocess(codes_fn, bpe_vocab_fn)

        vocab_path = os.path.expandvars(config['inference']['h5']['vocab'])
        with h5py.File(vocab_path, 'r') as h5_file:
            self.vocab = h5_file['vocab'][:]
            if 'additional_special_tokens' in h5_file['vocab'].attrs:
                additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
                vocab_special_chars = self.vocab[5:5 + additional_special_tokens].tolist()
            else:
                vocab_special_chars = []

        ft_corpus_path = os.path.expandvars(config['inference']['h5']['ft_corpus'])
        self.cl = StreamingH5CorpusLoader.load_and_split(
            ft_corpus_path,
            use_split_id=config['inference']['h5']['ft_corpus_split'],
            forced_vocab=(self.vocab, vocab_special_chars)
        )[0]

        pretrained_mdl_path = os.path.expandvars(config['inference']['pretrained_model'])
        self.load_model(len(self.vocab), None, pretrained_mdl_path)
        self._model.eval()

        self.source_input_fn = os.path.expandvars(config['inference']['source_fn'])
        self.hyp_output_fn = os.path.expandvars(config['inference']['hyp_fn'])
        if config['inference']['output_buffering']:
            self.buffering = -1
        else:
            self.buffering = 1

    def _infer(self, line: str):
        return self.infer(
            self.cl,
            line,
            self.model,
            self.config['inference']['beam_width'],
            self.config['inference']['max_len_scale'],
            self.config['inference']['noising_beta'],
            self.config['inference']['temperature'],
            self.device
        )

    def run(self):
        with open(self.source_input_fn, 'r') as in_f:
            with open(self.hyp_output_fn, 'w', buffering=self.buffering) as out_f:
                for li, line in enumerate(in_f):
                    if li < self.config['inference']['line_offset']:
                        continue
                    line = line.strip()
                    if self.config['inference']['preprocess']:
                        line = self.spacy_bpe_pp(line)
                    print('IN  : {}'.format(line))
                    if self.config['inference']['max_len'] > 0 and len(line.split(' ')) > self.config['inference']['max_len']:
                        print('TOO LONG')
                        continue

                    decoded_sentence = self._infer(line)

                    if self.config['inference']['remove_bpe_placeholder']:
                        decoded_sentence = decoded_sentence.replace("@@ ", "")
                    if self.config['inference']['output_parallel']:
                        out_f.write('{} <split> {}\n'.format(line, decoded_sentence))
                    else:
                        out_f.write('{}\n'.format(decoded_sentence))
                    print('OUT : {}'.format(decoded_sentence))
