import os
import h5py
from stylecorrection.operation.common import *
from stylecorrection.loaders.corpus import *


class PretrainOperation(TrainableOperation):

    def __init__(self,
                 global_conf: GlobalConfig,
                 eval_conf: EvaluationConfig,
                 model_conf: TransformerConfig,
                 pretrain_conf: PretrainConfig,
                 optimizer_conf: OptimizerConfig,
                 device: str):
        super(PretrainOperation, self).__init__(
            global_conf,
            eval_conf,
            device,
            model_conf=model_conf,
            optimizer_conf=optimizer_conf,
            hd5_dataset_config=pretrain_conf.hd5_dataset
        )
        self.pretrain_conf = pretrain_conf
        self.best_valid_loss = float('inf')
        self.current_training_step = 0

        self.load_dataset(pretrain_conf.max_sent_len)

        if pretrain_conf.algo == PretrainAlgo.BART:
            print('Using text infilling (BART)')
            self.pds_train = StreamingBARTPretrainingDataset(
                self.cl_train,
                tokens_per_batch=pretrain_conf.tpb,
                max_trainable_tokens=pretrain_conf.mttpb,
                device=device
            )
            self.pds_valid = StreamingBARTPretrainingDataset(
                self.cl_valid,
                tokens_per_batch=pretrain_conf.tpb,
                max_trainable_tokens=pretrain_conf.mttpb,
                device=device
            )
        elif pretrain_conf.algo == PretrainAlgo.MASS:
            self.pds_train = StreamingMASSPretrainingDataset(
                self.cl_train,
                tokens_per_batch=pretrain_conf.tpb,
                max_trainable_tokens=pretrain_conf.mttpb,
                device=device
            )
            self.pds_valid = StreamingMASSPretrainingDataset(
                self.cl_valid,
                tokens_per_batch=pretrain_conf.tpb,
                max_trainable_tokens=pretrain_conf.mttpb,
                device=device
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.cl_train.pad_idx)
        self.load_model(len(self.vocab), self.criterion)
        self.load_optimizer(pretrain_conf.optimizer)
        self.load_lr_scheduler()
        self.load_checkpoint()

    def run(self):
        print('Starting Pretraining...')
        train_losses = []
        max_reached = False

        for i in range(self.pretrain_conf.training_max.amount):
            if max_reached:
                break
            self.model.train()
            if self.cl_train.group_indexing:
                current_groups_offsets = self.cl_train.group_offsets.clone()
            else:
                current_groups_offsets = None
            for tbi, (t_enc_in, t_enc_in_key_mask, t_dec_out, t_dec_in, t_dec_in_key_mask, t_offsets) in enumerate(
                    self.pds_train):
                if self.pretrain_conf.training_max.use == TrainingMax.STEPS and self.current_training_step >= \
                        self.pretrain_conf.training_max.amount:
                    print('Max steps reached.')
                    max_reached = True
                if tbi % self.eval_conf.interval == 0 or max_reached:
                    self.model.eval()
                    valid_losses = []
                    with torch.no_grad():
                        for vbi, (v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets) \
                                in enumerate(self.pds_valid):
                            if self.in_multigpu_mode:
                                loss = self.model(
                                    v_dec_out,
                                    v_enc_in,
                                    v_dec_in,
                                    v_enc_in_key_mask,
                                    v_dec_in_key_mask,
                                    v_offsets
                                )
                                loss = loss.mean()
                            else:
                                out = self.model(
                                    v_enc_in,
                                    v_dec_in,
                                    v_enc_in_key_mask,
                                    v_dec_in_key_mask,
                                    v_offsets
                                )
                                loss = self.criterion(
                                    out.contiguous().view(-1, len(self.cl_valid.vocab)),
                                    v_dec_out.view(-1)
                                )
                            valid_losses.append(loss.item())
                            if vbi == self.eval_conf.num_valid_batch:
                                break
                        v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets = next(
                            iter(self.pds_valid))

                        if self.pretrain_conf.algo == PretrainAlgo.BART:
                            out = self.model(
                                v_enc_in[:1],
                                v_dec_in[:1],
                                v_enc_in_key_mask[:1],
                                v_dec_in_key_mask[:1],
                                None
                            )
                        else:
                            out = self.model(
                                v_enc_in[:1],
                                v_dec_in[:1],
                                v_enc_in_key_mask[:1],
                                v_dec_in_key_mask[:1],
                                v_offsets[:1]
                            )
                        if out.numel() == 0:
                            print(v_enc_in)
                        else:
                            enc_input = self.cl_train.decode_tensor(v_enc_in[:1])
                            expected_output = self.cl_train.decode_tensor(v_dec_out[:1])
                            predicted_output = self.cl_train.decode_tensor(out.argmax(dim=2))

                            print()
                            print('Masked sequence : {}'.format(enc_input))
                            print('Expected segment : {}'.format(expected_output))
                            print('Predicted segment: {}'.format(predicted_output))
                            print()

                        train_loss_mean = torch.tensor(train_losses).mean()
                        valid_loss_mean = torch.tensor(valid_losses).mean()
                        print('{}: Sentences Processed {}/{} / Num. Steps {} : Train:{:.4f}, Valid:{:.4f}'.format(
                            i,
                            self.cl_train.current_iterating_idx - t_enc_in.shape[0],
                            len(self.cl_train),
                            self.current_training_step,
                            train_loss_mean,
                            valid_loss_mean)
                        )

                        if valid_loss_mean < self.best_valid_loss:
                            self.best_valid_loss = valid_loss_mean
                            self.save_checkpoint(
                                Checkpoints.BEST,
                                current_groups_offsets,
                                t_enc_in.shape[0]
                            )

                        self.save_checkpoint(
                            Checkpoints.CURRENT,
                            current_groups_offsets,
                            t_enc_in.shape[0]
                        )

                    train_losses.clear()
                    self.model.train()

                if not max_reached:
                    self.optimizer.zero_grad()
                    out = self.model(t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
                    loss = self.criterion(out.contiguous().view(-1, len(self.cl_train.vocab)), t_dec_out.view(-1))
                    loss.backward()
                    if self.optimizer_conf.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_conf.grad_clip_norm)
                    train_losses.append(loss.item())
                    self.optimizer.step()
                    self.current_training_step += 1
                    current_groups_offsets = self.cl_train.group_offsets.clone()
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    break

        print('DONE')


