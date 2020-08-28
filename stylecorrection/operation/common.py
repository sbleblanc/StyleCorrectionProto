import torch
from enum import Enum
from typing import Callable
import torch.optim as optim
from stylecorrection.models.transformer import *
from stylecorrection.models.wrappers import *
from stylecorrection.utils.config import *
from stylecorrection.loaders.corpus import StreamingH5CorpusLoader
from abc import ABC, abstractmethod


class ModelNotLoadedException(Exception):

    def __init__(self):
        super(ModelNotLoadedException, self).__init__('load_model must be called first.')


class OptimizerNotLoaded(Exception):
    
    def __init__(self):
        super(OptimizerNotLoaded, self).__init__("load_optimizer must be called first.")


class LrSchedulerNotLoaded(Exception):

    def __init__(self):
        super(LrSchedulerNotLoaded, self).__init__("load_optimizer must be called first.")


class ConfigNotProvidedException(Exception):
    pass


class Checkpoints(Enum):
    CURRENT = 1
    BEST = 2


class Operation(ABC):

    def __init__(self,
                 global_conf: GlobalConfig,
                 device: str,
                 model_conf: TransformerConfig = None,
                 hd5_dataset_config: Hd5DatasetConfig = None):
        self.global_conf = global_conf
        self.model_conf = model_conf
        self.hd5_dataset_config = hd5_dataset_config
        self.device = device
        self.in_multigpu_mode = False
        self._model = None
        self.optimizer = None
        self.scheduler = None
        self.cl_train = None
        self.cl_valid = None
        self.vocab = None
        self.vocab_special_chars = None
        self.variables_to_save = []

    @property
    def model(self):
        if self.in_multigpu_mode:
            return self._model.module.model
        else:
            return self._model

    @abstractmethod
    def run(self):
        pass

    def load_model(self, len_vocab: int, criterion: Callable, pretrained_fn: str = None):
        if not self.model_conf:
            raise ConfigNotProvidedException('No model config provided')

        self._model = TransformerS2S(
            len_vocab,
            self.model_conf.emb_dim,
            self.model_conf.n_heads,
            self.model_conf.ff_dim,
            self.model_conf.num_enc_layers,
            self.model_conf.num_dec_layers,
            self.model_conf.activation
        )

        if self.global_conf.multi_gpu and torch.cuda.device_count() > 1:
            self.in_multigpu_mode = True
            self._model = DataParallelCELWrapper(self._model, criterion, len_vocab)
            self._model = nn.DataParallel(self._model)

        if pretrained_fn:
            with open(pretrained_fn, 'rb') as in_file:
                loaded_data = torch.load(in_file, map_location=self.device)
                self.model.load_state_dict(loaded_data['model_state_dict'])

        self.model.to(self.device)

    def load_dataset(self, max_sent_len: int = 0):
        if not self.hd5_dataset_config:
            raise ConfigNotProvidedException('No dataset config provided')

        corpus_h5_fn = os.path.expandvars(self.hd5_dataset_config.h5_fn)
        vocab_h5_fn = os.path.expandvars(self.hd5_dataset_config.vocab_fn)

        with h5py.File(vocab_h5_fn, 'r') as h5_file:
            self.vocab = h5_file['vocab'][:]
            if 'additional_special_tokens' in h5_file['vocab'].attrs:
                additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
                self.vocab_special_chars = self.vocab[5:5 + additional_special_tokens].tolist()
            else:
                self.vocab_special_chars = []

        self.cl_train, self.cl_valid = StreamingH5CorpusLoader.load_and_split(
            corpus_h5_fn,
            use_split_id=self.hd5_dataset_config.valid_split_id,
            forced_vocab=(self.vocab, self.vocab_special_chars),
            max_sent_len=max_sent_len,
            group_by_len=self.hd5_dataset_config.group_by_len,
            smoothing_alpha=self.hd5_dataset_config.smoothing_alpha,
            device=self.device
        )


class TrainableOperation(Operation):

    def __init__(self,
                 global_conf: GlobalConfig,
                 eval_config: EvaluationConfig,
                 device: str,
                 model_conf: TransformerConfig = None,
                 optimizer_conf: OptimizerConfig = None,
                 hd5_dataset_config: Hd5DatasetConfig = None,
                 model_files_config: ModelFilenameConfig = None):
        super(TrainableOperation, self).__init__(
            global_conf,
            device,
            model_conf=model_conf,
            hd5_dataset_config=hd5_dataset_config
        )
        self.eval_conf = eval_config
        self.optimizer_conf = optimizer_conf
        self.model_files_config = model_files_config
        self.best_valid_loss = float('inf')
        self.current_training_step = 0

    @abstractmethod
    def run(self):
        pass

    def load_optimizer(self, which: Optimizers):
        if self.model is None:
            raise ModelNotLoadedException

        if not self.optimizer_conf:
            raise ConfigNotProvidedException('No optimizer config provided')

        if which == Optimizers.ADAM:
            self.optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.optimizer_conf.adam.lr,
                                   betas=(self.optimizer_conf.adam.beta_1, self.optimizer_conf.adam.beta_2),
                                   eps=self.optimizer_conf.adam.eps,
                                   weight_decay=self.optimizer_conf.adam.weight_decay)
        elif which == Optimizers.SGD:
            self.optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.optimizer_conf.sgd.lr,
                                  momentum=self.optimizer_conf.sgd.momentum,
                                  weight_decay=self.optimizer_conf.sgd.weight_decay,
                                  nesterov=self.optimizer_conf.sgd.nesterov)

    def load_lr_scheduler(self):
        if self.optimizer is None:
            raise OptimizerNotLoaded

        if not self.optimizer_conf.scheduler:
            raise ConfigNotProvidedException('No learning rate scheduler config provided')

        if self.optimizer_conf.scheduler.use == LRSchedulers.ONE_CYCLE:
            pct = self.optimizer_conf.scheduler.one_cycle.warmup_steps / \
                  self.optimizer_conf.scheduler.one_cycle.total_step
            print('Scheduler Pct: {:%}'.format(pct))

            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.optimizer_conf.scheduler.one_cycle.max_lr,
                div_factor=self.optimizer_conf.scheduler.one_cycle.initial_lr_div,
                final_div_factor=self.optimizer_conf.scheduler.one_cycle.final_lr_div,
                total_steps=self.optimizer_conf.scheduler.one_cycle.total_step,
                anneal_strategy=self.optimizer_conf.scheduler.one_cycle.anneal_strategy,
                pct_start=pct,
                last_epoch=-1,
                cycle_momentum=False
            )

    def load_checkpoint(self):

        if self.model is None:
            raise ModelNotLoadedException

        if self.optimizer is None:
            raise OptimizerNotLoaded

        if self.model_files_config.resume_from == ResumeFrom.BEST:
            model_save_fn = os.path.expandvars(self.model_files_config.best_model)
        else:
            model_save_fn = os.path.expandvars(self.model_files_config.current_model)

        if os.path.exists(model_save_fn):
            with open(model_save_fn, 'rb') as data_file:
                print('Loading from {}'.format(model_save_fn))
                loaded_data = torch.load(data_file, map_location='cpu')
                if self.cl_train.group_indexing:
                    self.cl_train.state = {
                        k: loaded_data[k] for k in
                        ['current_group', 'current_group_selector', 'current_group_offsets'] if k in loaded_data
                    }
                else:
                    self.cl_train.state = {
                        k: loaded_data[k] for k in
                        ['current_iterating_idx', 'current_iterating_order'] if k in loaded_data
                    }
                self.cl_train.generate_iterating_order = False
                self.optimizer.load_state_dict(loaded_data['optim_state_dict'])
                self.model.load_state_dict(loaded_data['model_state_dict'])
                if 'best_valid_loss' in loaded_data:
                    self.best_valid_loss = loaded_data['best_valid_loss']
                if 'current_training_step' in loaded_data:
                    self.current_training_step = loaded_data['current_training_step']
                if self.optimizer_conf.scheduler == LRSchedulers.ONE_CYCLE:
                    if not self.scheduler:
                        raise LrSchedulerNotLoaded
                    self.scheduler.load_state_dict(loaded_data['scheduler_state_dict'])

    def save_checkpoint(self,
                        which: Checkpoints,
                        current_groups_offsets=None,
                        current_iterating_idx_offset: int = None):
        if which == Checkpoints.BEST:
            save_fn = os.path.expandvars(self.model_files_config.best_model)
        else:
            save_fn = os.path.expandvars(self.model_files_config.current_model)
        with open(save_fn, 'wb') as out_file:
            to_save = {
                'optim_state_dict': self.optimizer.state_dict(),
                'best_valid_loss': self.best_valid_loss,
                'current_training_step': self.current_training_step
            }
            to_save.update(self.cl_train.state)
            if self.cl_train.group_indexing:
                to_save['current_group_offsets'] = current_groups_offsets
            else:
                to_save['current_iterating_idx'] -= current_iterating_idx_offset
            to_save['model_state_dict'] = self.model.state_dict()
            if self.scheduler:
                to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(to_save, out_file)
