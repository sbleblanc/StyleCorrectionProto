import torch
from enum import Enum
from typing import Callable
import torch.optim as optim
from stylecorrection.models.transformer import *
from stylecorrection.models.wrappers import *


class Optimizers(Enum):
    ADAM = 1
    SGD = 2

class LRSchedulers(Enum):
    ONE_CYCLE = 1

class ModelNotLoadedException(Exception):

    def __init__(self):
        super(ModelNotLoadedException, self).__init__('load_model must be called first.')


class OptimizerNotLoaded(Exception):
    
    def __init__(self):
        super(OptimizerNotLoaded, self).__init__("load_optimizer must be called first.")


class Operation(object):

    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device
        self._model = None
        self.in_multigpu_mode = False
        self.optimizer = None
        self.scheduler = None
        self.variables_to_save = []

    @property
    def model(self):
        if self.in_multigpu_mode:
            return self._model.module.model
        else:
            return self._model

    def load_model(self, len_vocab: int, criterion: Callable, pretrained_fn: str = None):
        self._model = TransformerS2S(
            len_vocab,
            self.config['TransformerS2S']['emb_dim'],
            self.config['TransformerS2S']['n_head'],
            self.config['TransformerS2S']['ff_dim'],
            self.config['TransformerS2S']['num_enc_layers'],
            self.config['TransformerS2S']['num_dec_layers'],
            self.config['TransformerS2S']['activation']
        )

        if self.config['multi_gpu'] and torch.cuda.device_count() > 1:
            self.in_multigpu_mode = True
            self._model = DataParallelCELWrapper(self._model, criterion, len_vocab)
            self._model = nn.DataParallel(self._model)

        if pretrained_fn:
            with open(pretrained_fn, 'rb') as in_file:
                loaded_data = torch.load(in_file, map_location=self.device)
                self.model.load_state_dict(loaded_data['model_state_dict'])

        self.model.to(self.device)

    def load_optimizer(self, which: Optimizers):
        if self.model is None:
            raise ModelNotLoadedException
        if which == Optimizers.ADAM:
            self.optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.config['optimizer']['adam']['lr'],
                                   betas=(self.config['optimizer']['adam']['beta_1'], self.config['optimizer']['adam']['beta_2']),
                                   eps=self.config['optimizer']['adam']['eps'],
                                   weight_decay=self.config['optimizer']['adam']['weight_decay'])
        elif which == Optimizers.SGD:
            self.optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.config['optimizer']['sgd']['lr'],
                                  momentum=self.config['optimizer']['sgd']['momentum'],
                                  weight_decay=self.config['optimizer']['sgd']['weight_decay'],
                                  nesterov=self.config['optimizer']['sgd']['nesterov'])

    def load_lr_scheduler(self, which: LRSchedulers):
        if self.optimizer is None:
            raise OptimizerNotLoaded

        if which == LRSchedulers.ONE_CYCLE:
            pct = self.config['optimizer']['scheduler']['one_cycle']['warmup_steps'] / \
                  self.config['optimizer']['scheduler']['one_cycle']['total_steps']
            print('Scheduler Pct: {:%}'.format(pct))

            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.config['optimizer']['scheduler']['one_cycle']['max_lr'],
                div_factor=self.config['optimizer']['scheduler']['one_cycle']['initial_lr_div'],
                final_div_factor=self.config['optimizer']['scheduler']['one_cycle']['final_lr_div'],
                total_steps=self.config['optimizer']['scheduler']['one_cycle']['total_steps'],
                anneal_strategy=self.config['optimizer']['scheduler']['one_cycle']['anneal_strategy'],
                pct_start=pct,
                last_epoch=-1,
                cycle_momentum=False
            )
