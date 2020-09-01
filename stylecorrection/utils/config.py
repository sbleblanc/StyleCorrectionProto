import inspect
import os
import h5py
from enum import Enum
from typing import List, Optional


class TrainingMax(Enum):
    STEPS = 1
    EPOCH = 2


class ResumeFrom(Enum):
    CURRENT = 1
    BEST = 2


class OperationModes(Enum):
    HD5_GEN = 1
    GEN_SPLIT = 2
    PRETRAIN = 3
    FINETUNE = 4
    INFERENCE = 5
    EVAL = 6
    DEBUG = 7
    VOCAB_DUMP = 8


class Optimizers(Enum):
    ADAM = 1
    SGD = 2


class LRSchedulers(Enum):
    NONE = 1
    ONE_CYCLE = 2


class PretrainAlgo(Enum):
    MASS = 1
    BART = 2


class FreezeOptions(Enum):
    NOTHING = 1
    EMB = 2
    EMB_ENC = 3
    EMB_ENC_DEC = 4


class BestCriteria(Enum):
    VALID_LOSS = 1
    GLEU = 2


class ShufflerType(Enum):
    NORMAL_NOISE = 1
    CHUNK_SWAP = 2


class FinetuneDatasets(Enum):
    PARALLEL = 1
    CA = 2


class BaseConfig(object):
    
    def __init__(self, local_values):
        signature = inspect.signature(self.__init__)
        for p in signature.parameters:
            if signature.parameters[p].annotation:
                if not isinstance(local_values[p], signature.parameters[p].annotation):
                    raise TypeError('({}) {} must be of type {} but current value is {}'.format(
                        local_values['self'],
                        p,
                        signature.parameters[p].annotation,
                        local_values[p])
                    )

    def get_attr_value(self, attr: str):
        return self.__dict__[attr]

    @classmethod
    def from_yaml(cls, loader, node):
        params = loader.construct_mapping(node)
        return cls(**params)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = dict([(key, data.get_attr_value(key)) for key in data.__dict__.keys()])
        return dumper.represent_mapping(data.yaml_tag, mapping)

class TransformerConfig(BaseConfig):

    @property
    def yaml_tag(self):
        return '!TransformerS2S'

    def __init__(self,
                 emb_dim: int = 768,
                 n_head: int = 8,
                 ff_dim: int = 4096,
                 num_enc_layers: int = 6,
                 num_dec_layers: int = 6,
                 activation: str = 'relu'):
        super(TransformerConfig, self).__init__(locals())

        if activation not in ['relu', 'gelu']:
            raise ValueError("(TransformerS2S) activation can only be 'relu' ou 'gelu'")

        self.emb_dim = emb_dim
        self.n_heads = n_head
        self.ff_dim = ff_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.activation = activation


class EvaluationConfig(BaseConfig):

    @property
    def yaml_tag(self):
        return '!eval'

    def __init__(self,
                 interval: int = 250,
                 num_valid_batch: int = 100):
        super(EvaluationConfig, self).__init__(locals())
        self.interval = interval
        self.num_valid_batch = num_valid_batch


class GenerateHd5Config(BaseConfig):

    def __init__(self,
                 h5_fn: str,
                 corpus_tar_gz: str,
                 topk: int = 0,
                 max_len: int = 9999999,
                 additional_tokens: list = [],
                 valid_ratio: float = 0.8):
        super(GenerateHd5Config, self).__init__(locals())
        self.h5_fn = h5_fn
        self.corpus_tar_gz = corpus_tar_gz
        self.topk = topk
        self.max_len = max_len
        self.additional_tokens = additional_tokens
        if valid_ratio <= 1.:
            self.valid_ratio = valid_ratio
        else:
            self.valid_ratio = int(valid_ratio)


class TrainingMaxConfig(BaseConfig):

    @property
    def yaml_tag(self):
        return '!training_max'

    def __init__(self,
                 use: str,
                 amount: int):
        super(TrainingMaxConfig, self).__init__(locals())

        if use == 'steps':
            self.use = TrainingMax.STEPS
        elif use == 'epoch':
            self.use = TrainingMax.EPOCH
        else:
            raise ValueError("(TrainingMax) use can only be 'steps' or 'epoch'")

        self.amount = amount

    def get_attr_value(self, attr: str):
        if attr == 'use':
            if self.use == TrainingMax.STEPS:
                return 'steps'
            elif self.use == TrainingMax.EPOCH:
                return 'epoch'
        else:
            return BaseConfig.get_attr_value(self, attr)

class PreprocessConfig(BaseConfig):

    def __init__(self,
                 bpe_codes_fn: str,
                 bpe_vocab_fn: str):
        super(PreprocessConfig, self).__init__(locals())
        self.bpe_codes_fn = bpe_codes_fn
        self.bpe_vocab_fn = bpe_vocab_fn


class ModelFilenameConfig(BaseConfig):

    def __init__(self,
                 best_model: str,
                 current_model: str,
                 resume_from: str):
        super(ModelFilenameConfig, self).__init__(locals())

        if resume_from == 'current':
            self.resume_from = ResumeFrom.CURRENT
        elif resume_from == 'best':
            self.resume_from = ResumeFrom.BEST
        else:
            raise ValueError("(ModelFilename) resume_from can only be 'current' or 'best'")

        self.best_model = best_model
        self.current_model = current_model


class Hd5DatasetConfig(BaseConfig):

    def __init__(self,
                 h5_fn: str,
                 vocab_fn: str,
                 valid_split_id: int = 0,
                 smoothing_alpha: int = 1,
                 group_by_len: bool = False):
        super(Hd5DatasetConfig, self).__init__(locals())

        self.h5_fn = h5_fn
        self.vocab_fn = vocab_fn
        self.valid_split_id = valid_split_id
        self.smoothing_alpha = smoothing_alpha
        self.group_by_len = group_by_len


class GlobalConfig(BaseConfig):

    def __init__(self,
                 mode: str,
                 multi_gpu: bool = False):
        super(GlobalConfig, self).__init__(locals())

        if mode == 'hd5_gen':
            self.mode = OperationModes.HD5_GEN
        elif mode == 'gen_split':
            self.mode = OperationModes.GEN_SPLIT
        elif mode == 'pretrain':
            self.mode = OperationModes.PRETRAIN
        elif mode == 'finetune':
            self.mode = OperationModes.FINETUNE
        elif mode == 'eval':
            self.mode = OperationModes.EVAL
        elif mode == 'debug':
            self.mode = OperationModes.DEBUG
        elif mode == 'inference':
            self.mode = OperationModes.INFERENCE
        elif mode == 'vocab_dump':
            self.mode = OperationModes.VOCAB_DUMP
        else:
            raise ValueError('(Golbal) mode has to be in [{}]'.format(
                ', '.join(['hd5_gen', 'gen_split', 'finetune', 'eval', 'debug', 'inference', 'vocab_dump'])
            ))

        self.multi_gpu = multi_gpu


class OneCycleConfig(BaseConfig):

    @property
    def yaml_tag(self):
        return '!one_cycle'

    def __init__(self,
                 max_lr: float,
                 initial_lr_div: float,
                 final_lr_div: float,
                 warmup_steps: int,
                 total_steps: int,
                 base_momentum: float = 0.,
                 max_momentum: float = 0.,
                 anneal_strategy: str = 'linear'):
        super(OneCycleConfig, self).__init__(locals())

        self.max_lr = max_lr
        self.initial_lr_div = initial_lr_div
        self.final_lr_div = final_lr_div
        self.warmup_steps = warmup_steps
        self.total_step = total_steps
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.anneal_strategy = anneal_strategy


class AdamConfig(BaseConfig):

    def __init__(self,
                 lr: float,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        super(AdamConfig, self).__init__(locals())

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay


class SgdConfig(BaseConfig):

    def __init__(self,
                 lr: float,
                 momentum: float = 0.99,
                 weight_decay: float = 0.0,
                 nesterov: bool = False):
        super(SgdConfig, self).__init__(locals())
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov


class SchedulerConfig(BaseConfig):

    @property
    def yaml_tag(self):
        return '!scheduler'

    def __init__(self,
                 use: str,
                 one_cycle: OneCycleConfig):
        super(SchedulerConfig, self).__init__(locals())

        if use == 'one_cycle':
            self.use = LRSchedulers.ONE_CYCLE
        elif use == 'none':
            self.use = LRSchedulers.NONE
        else:
            raise ValueError('(Scheduler) use can only be one_cycle or none')
        self.one_cycle = one_cycle

    def get_attr_value(self, attr: str):
        if attr == 'use':
            if self.use == LRSchedulers.ONE_CYCLE:
                return 'steps'
            elif self.use == LRSchedulers.NONE:
                return 'epoch'
        else:
            return BaseConfig.get_attr_value(self, attr)


class OptimizerConfig(BaseConfig):

    def __init__(self,
                 grad_clip_norm: float,
                 scheduler: SchedulerConfig,
                 adam: AdamConfig,
                 sgd: SgdConfig):
        super(OptimizerConfig, self).__init__(locals())
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler
        self.adam = adam
        self.sgd = sgd


class InferenceConfig(BaseConfig):

    def __init__(self,
                 hd5_dataset: Hd5DatasetConfig,
                 pretrained_model: str,
                 force_cpu: bool,
                 beam_width: int,
                 source_fn: str,
                 hyp_fn: str,
                 max_len: int,
                 max_len_scale: float,
                 remove_bpe_placeholder: bool,
                 output_parallel: bool,
                 preprocess: bool,
                 output_buffering: bool = False,
                 temperature: float = 1.,
                 line_offset: int = 0,
                 noising_beta: float = 0.,):
        super(InferenceConfig, self).__init__(locals())
        self.hd5_dataset = hd5_dataset
        self.pretrained_model = pretrained_model
        self.force_cpu = force_cpu
        self.beam_width = beam_width
        self.source_fn = source_fn
        self.hyp_fn = hyp_fn
        self.max_len = max_len
        self.max_len_scale = max_len_scale
        self.remove_bpe_placeholder = remove_bpe_placeholder
        self.output_parallel = output_parallel
        self.preprocess = preprocess
        self.output_buffering = output_buffering
        self.temperature = temperature
        self.line_offset = line_offset
        self.noising_beta = noising_beta


class VocabDumpConfig(BaseConfig):

    def __init__(self,
                 hd5_dataset: Hd5DatasetConfig,
                 vocab_topk: int,
                 min_freq: int):
        super(VocabDumpConfig, self).__init__(locals())
        self.hd5_dataset = hd5_dataset
        self.vocab_topk = vocab_topk
        self.min_freq = min_freq


class PretrainConfig(BaseConfig):

    def __init__(self,
                 hd5_dataset: Hd5DatasetConfig,
                 optimizer: str,
                 max_sent_len: int,
                 algo: str,
                 training_max: TrainingMaxConfig,
                 model_files: ModelFilenameConfig,
                 tpb: int,
                 mttpb: int):
        super(PretrainConfig, self).__init__(locals())

        if optimizer == 'adam':
            self.optimizer = Optimizers.ADAM
        elif optimizer == 'sgd':
            self.optimizer = Optimizers.SGD
        else:
            raise ValueError('(Pretrain) optimizer can only be adam or sgd')

        self.hd5_dataset = hd5_dataset
        self.max_sent_len = max_sent_len
        self.algo = algo
        self.training_max = training_max
        self.model_files = model_files
        self.tpb = tpb
        self.mttpb = mttpb

class GleuConfig(BaseConfig):

    def __init__(self,
                 refs: list,
                 src: str,
                 n: int = 4,
                 iter: int = 500,
                 sent: bool = False,
                 preprocess: bool = True):
        super(GleuConfig, self).__init__(locals())
        self.refs = refs
        self.src = src
        self.n = n
        self.iter = iter
        self.sent = sent
        self.preprocess = preprocess


class ParallelDatasetConfig(BaseConfig):

    def __init__(self,
                 reverse: bool = False,
                 split_token: str = '<split>'):
        super(ParallelDatasetConfig, self).__init__(locals())
        self.reverse = reverse
        self.split_token = split_token


class CaDatasetConfig(BaseConfig):

    def __init__(self,
                 replace_prob: float = 0.0,
                 del_prob: float = 0.0,
                 ins_prob: float = 0.0,
                 keep_prob: float = 1.0,
                 mask_prob: float = 0.0,
                 shuffle_prob: float = 0.0,
                 shuffler: str = 'normal',
                 sigma: float = 0.0,
                 min_chunk_ratio: float = 0.3,
                 max_chunk_ratio: float = 0.5):
        super(CaDatasetConfig, self).__init__(locals())

        if shuffler == 'chunk':
            self.shuffler = ShufflerType.CHUNK_SWAP
        elif shuffler == 'normal':
            self.shuffler = ShufflerType.NORMAL_NOISE
        else:
            raise ValueError('(CADataset) shuffler can only be chunk or normal')

        self.replace_prob = replace_prob
        self.del_prob = del_prob
        self.ins_prob = ins_prob
        self.keep_prob = keep_prob
        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob
        self.sigma = sigma
        self.min_chunk_ratio = min_chunk_ratio
        self.max_chunk_ratio = max_chunk_ratio


class FinetuneDatasetConfig(BaseConfig):

    def __init__(self,
                 tpb: int,
                 to_use: str = 'ca',
                 parallel: ParallelDatasetConfig = ParallelDatasetConfig(),
                 ca: CaDatasetConfig = CaDatasetConfig()):
        super(FinetuneDatasetConfig, self).__init__(locals())

        self.to_use: List[FinetuneDatasets] = []

        if to_use == 'ca':
            self.to_use.append(FinetuneDatasets.CA)
        elif to_use == 'parallel':
            self.to_use.append(FinetuneDatasets.PARALLEL)
        elif '+' in to_use:
                ds_split = to_use.split('+')
                for ds in ds_split:
                    self.to_use.append(FinetuneDatasets[ds])
        else:
            raise ValueError('(FinetuneDataset) to_use can only be parallel or ca or a combination')

        self.tpb = tpb
        self.parallel = parallel
        self.ca = ca


class FinetuneConfig(BaseConfig):

    def __init__(self,
                 optimizer: str,
                 model_files: ModelFilenameConfig,
                 pretrain_model_fn: str,
                 max_sent_len: int,
                 freeze: str,
                 best_criteria: str,
                 training_max: TrainingMaxConfig,
                 dataset: FinetuneDatasetConfig,
                 hd5_dataset: Hd5DatasetConfig):
        super(FinetuneConfig, self).__init__(locals())
        if optimizer == 'adam':
            self.optimizer = Optimizers.ADAM
        elif optimizer == 'sgd':
            self.optimizer = Optimizers.SGD
        else:
            raise ValueError('(Pretrain) optimizer can only be adam or sgd')

        if freeze == 'None':
            self.freeze = FreezeOptions.NOTHING
        elif freeze == 'emb':
            self.freeze = FreezeOptions.EMB
        elif freeze == 'enc':
            self.freeze = FreezeOptions.EMB_ENC
        elif freeze == 'encdec':
            self.freeze = FreezeOptions.EMB_ENC_DEC
        else:
            raise ValueError('(Pretrain) freeze can only be None, emb, enc or encdec')

        if best_criteria == 'valid':
            self.best_criteria = BestCriteria.VALID_LOSS
        elif best_criteria == 'gleu':
            self.best_criteria = BestCriteria.GLEU
        else:
            raise ValueError('(Pretrain) best_criteria can only be valid of gleu')

        self.pretrain_model_fn = pretrain_model_fn
        self.model_files = model_files
        self.max_sent_len = max_sent_len
        self.training_max = training_max
        self.dataset = dataset
        self.hd5_dataset = hd5_dataset


class SampleCorrectionsConfig(BaseConfig):

    def __init__(self,
                 dirty: list,
                 clean: list):
        super(SampleCorrectionsConfig, self).__init__(locals())

        if len(dirty) != len(clean):
            raise ValueError('(SampleCorrections) the number of dirty sentences must match the number of clean ones')

        self.dirty = dirty
        self.clean = clean


class ManualEvalConfig(BaseConfig):

    def __init__(self,
                 hd5_dataset: Hd5DatasetConfig,
                 pretrained_model: str,
                 force_cpu: bool,
                 beam_width: int,
                 max_len_scale: float,
                 sample_corrections: SampleCorrectionsConfig):
        super(ManualEvalConfig, self).__init__(locals())
        self.hd5_dataset = hd5_dataset
        self.pretrained_model = pretrained_model
        self.force_cpu = force_cpu
        self.beam_width = beam_width
        self.max_len_scale = max_len_scale
        self.sample_corrections = sample_corrections


class AllConfig(BaseConfig):

    def __init__(self,
                 global_conf: GlobalConfig,
                 eval: EvaluationConfig,
                 transformer_s2s: TransformerConfig,
                 hd5_gen: GenerateHd5Config,
                 vocab_dump: VocabDumpConfig,
                 pretrain: PretrainConfig,
                 preprocess: PreprocessConfig,
                 gleu: GleuConfig,
                 finetune: FinetuneConfig,
                 optimizer: OptimizerConfig,
                 inference: InferenceConfig,
                 manual_eval: ManualEvalConfig):
        super(AllConfig, self).__init__(locals())

        self.global_conf = global_conf
        self.eval = eval
        self.transformer_s2s = transformer_s2s
        self.hd5_gen = hd5_gen
        self.vocab_dump = vocab_dump
        self.pretrain = pretrain
        self.preprocess = preprocess
        self.gleu = gleu
        self.finetune = finetune
        self.optimizer = optimizer
        self.inference = inference
        self.manual_eval = manual_eval
