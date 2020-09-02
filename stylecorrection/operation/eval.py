from stylecorrection.operation.common import Operation
from stylecorrection.utils.config import *
from stylecorrection.loaders.corpus import *
from stylecorrection.models.transformer import TransformerS2S


class ManualEvaluationOperation(Operation):

    def __init__(self,
                 global_conf: GlobalConfig,
                 transformer_s2s_conf: TransformerConfig,
                 man_eval_conf: ManualEvalConfig,
                 device: str):
        super(ManualEvaluationOperation, self).__init__(global_conf, device, transformer_s2s_conf)
        self.man_eval_conf = man_eval_conf
        self.transformer_s2s_conf = transformer_s2s_conf

        if man_eval_conf.force_cpu:
            self.device = "cpu"

        vocab_path = os.path.expandvars(man_eval_conf.hd5_dataset.vocab_fn)
        with h5py.File(vocab_path, 'r') as h5_file:
            vocab = h5_file['vocab'][:]
            if 'additional_special_tokens' in h5_file['vocab'].attrs:
                additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
                vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
            else:
                vocab_special_chars = []

        ft_corpus_path = os.path.expandvars(man_eval_conf.hd5_dataset.h5_fn)
        self.cl = StreamingH5CorpusLoader.load_and_split(
            ft_corpus_path,
            use_split_id=man_eval_conf.hd5_dataset.valid_split_id,
            forced_vocab=(vocab, vocab_special_chars)
        )[0]
        criterion_placeholder = nn.CrossEntropyLoss(ignore_index=self.cl_train.pad_idx)
        pretrained_mdl_path = os.path.expandvars(man_eval_conf.pretrained_model)
        self.load_model(len(vocab), criterion_placeholder, pretrained_fn=pretrained_mdl_path)

    def run(self):
        print('Starting manual evaluation...')
        for ds, cs in zip(self.man_eval_conf.sample_corrections.dirty,
                          self.man_eval_conf.sample_corrections.clean):
            test_sentence = self.cl.encode_sentence(ds).to(self.device)

            with torch.no_grad():
                res = self.model.beam_decode_2(
                    test_sentence,
                    torch.tensor([self.cl.bos_idx], dtype=torch.long),
                    beam_width=self.man_eval_conf.beam_width,
                    max_len=int(len(test_sentence) * self.man_eval_conf.max_len_scale),
                    end_token=self.cl.eos_idx,
                    return_scores=True,
                    device=self.device
                )

            print("IN: {}".format(ds))
            print("EXPECTED: {}".format(cs))
            for s, b in res:
                decoded = self.cl.decode_tensor(b)
                print('\t({:.4f}) : {}'.format(s, decoded[0]))
