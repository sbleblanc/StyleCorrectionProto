from stylecorrection.operation.common import Operation
from stylecorrection.loaders.corpus import *
from stylecorrection.utils.config import *


class Hd5GenOperation(Operation):

    def __init__(self,
                 global_conf: GlobalConfig,
                 hd5_config: GenerateHd5Config,
                 device: str,):
        super(Hd5GenOperation, self).__init__(global_conf, device)
        self.hd5_config = hd5_config

    def run(self):
        h5_fn = os.path.expandvars(self.hd5_config.h5_fn)
        if self.global_conf.mode == OperationModes.HD5_GEN:
            print('Creating HD5 dataset...')
            additional_tokens = []
            if self.hd5_config.additional_tokens is not None:
                additional_tokens = self.hd5_config.additional_tokens

            StreamingH5CorpusLoader.create_from_compressed(
                h5_fn,
                os.path.expandvars(self.hd5_config.corpus_tar_gz),
                lambda x: x.strip().split(' '),
                lambda x: x.replace('\x85', '').replace('\u2028', '').lower(),
                self.hd5_config.topk,
                self.hd5_config.max_len,
                additional_tokens
            )
            print('DONE')
        elif self.global_conf.mode == OperationModes.GEN_SPLIT:
            print('Generating train/valid split...')
            StreamingH5CorpusLoader.generate_split(
                h5_fn,
                self.hd5_config.valid_ratio
            )
            print('DONE')


class VocabDumpOperation(Operation):

    def __init__(self,
                 global_conf: GlobalConfig,
                 vocab_dump_config: VocabDumpConfig,
                 device: str):
        super(VocabDumpOperation, self).__init__(global_conf, device)
        self.vocab_dump_config = vocab_dump_config

    def run(self):
        print('Starting vocab dumping...')
        corpus_h5_fn = os.path.expandvars(self.vocab_dump_config.hd5_dataset.h5_fn)
        vocab_h5_fn = os.path.expandvars(self.vocab_dump_config.hd5_dataset.vocab_fn)
        cl = StreamingH5CorpusLoader.load_and_split(
            corpus_h5_fn,
            self.vocab_dump_config.hd5_dataset.valid_split_id,
            self.vocab_dump_config.vocab_topk,
            self.vocab_dump_config.min_freq,
            device=self.device
        )[0]
        cl.dump_vocab(vocab_h5_fn)
        print('DONE')
