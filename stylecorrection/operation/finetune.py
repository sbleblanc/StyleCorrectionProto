import os
import h5py
from stylecorrection.operation.common import *
from stylecorrection.operation.inference import InferenceOperation
from stylecorrection.loaders.corpus import *
from stylecorrection.utils.GLEU import GLEU


class FinetuneStreamingOperation(TrainableOperation):

    @classmethod
    def __get_finetune_dataset(cls,
                               which: FinetuneDatasets,
                               ds_conf: FinetuneDatasetConfig,
                               train_cl: StreamingH5CorpusLoader,
                               valid_cl: StreamingH5CorpusLoader,
                               device: str,):
        if which == FinetuneDatasets.CA:
            if ds_conf.ca.shuffler == ShufflerType.CHUNK_SWAP:
                shuffler = SentenceShuffler.chunk_shuffler(ds_conf.ca.min_chunk_ratio, ds_conf.ca.max_chunk_ratio)
            elif ds_conf.ca.shuffler == ShufflerType.NORMAL_NOISE:
                shuffler = SentenceShuffler.normal_shuffler(ds_conf.ca.sigma)

            dnds_train = StreamingCANoiseDataset(train_cl,
                                                 replace_prob=ds_conf.ca.replace_prob,
                                                 del_prob=ds_conf.ca.del_prob,
                                                 ins_prob=ds_conf.ca.ins_prob,
                                                 keep_prob=ds_conf.ca.keep_prob,
                                                 mask_prob=ds_conf.ca.mask_prob,
                                                 shuffle_prob=ds_conf.ca.shuffle_prob,
                                                 shuffler=shuffler,
                                                 tokens_per_batch=ds_conf.tpb,
                                                 max_trainable_tokens=ds_conf.tpb,
                                                 device=device)

            dnds_valid = StreamingCANoiseDataset(valid_cl,
                                                 replace_prob=ds_conf.ca.replace_prob,
                                                 del_prob=ds_conf.ca.del_prob,
                                                 ins_prob=ds_conf.ca.ins_prob,
                                                 keep_prob=ds_conf.ca.keep_prob,
                                                 mask_prob=ds_conf.ca.mask_prob,
                                                 shuffle_prob=ds_conf.ca.shuffle_prob,
                                                 shuffler=shuffler,
                                                 tokens_per_batch=ds_conf.tpb,
                                                 max_trainable_tokens=ds_conf.tpb,
                                                 device=device)

        elif which == FinetuneDatasets.PARALLEL:
            dnds_train = StreamingParallelDataset(train_cl,
                                                  split_token=ds_conf.parallel.split_token,
                                                  reverse=ds_conf.parallel.reverse,
                                                  tokens_per_batch=ds_conf.tpb,
                                                  max_trainable_tokens=ds_conf.tpb,
                                                  device=device)
            dnds_valid = StreamingParallelDataset(valid_cl,
                                                  split_token=ds_conf.parallel.split_token,
                                                  reverse=ds_conf.parallel.reverse,
                                                  tokens_per_batch=ds_conf.tpb,
                                                  max_trainable_tokens=ds_conf.tpb,
                                                  device=device)

        return dnds_train, dnds_valid

    def __init__(self,
                 global_conf: GlobalConfig,
                 eval_conf: EvaluationConfig,
                 finetune_conf: FinetuneConfig,
                 optimizer_conf: OptimizerConfig,
                 model_conf: TransformerConfig,
                 inference_conf: InferenceConfig,
                 device: str,
                 gleu_conf: GleuConfig = None,
                 preprocess_conf: PreprocessConfig = None):
        super(FinetuneStreamingOperation, self).__init__(
            global_conf,
            eval_conf,
            device,
            model_conf=model_conf,
            optimizer_conf=optimizer_conf,
            hd5_dataset_config=finetune_conf.hd5_dataset,
            model_files_config=finetune_conf.model_files
        )
        self.finetune_conf = finetune_conf
        self.inference_conf = inference_conf
        self.gleu_conf = gleu_conf
        self.preprocess_conf = preprocess_conf
        self.load_dataset(finetune_conf.max_sent_len)

        if len(finetune_conf.dataset.to_use) > 1:
            train_datasets = []
            valid_datasets = []
            for ds in finetune_conf.dataset.to_use:
                tds, vds = self.__get_finetune_dataset(
                    ds,
                    finetune_conf.dataset,
                    self.cl_train,
                    self.cl_valid
                )
                train_datasets.append(tds)
                valid_datasets.append(vds)

            self.dnds_train = StreamingChainedDataset(
                self.cl_train,
                train_datasets,
                finetune_conf.dataset.tpb,
                finetune_conf.dataset.tpb,
                device
            )

            self.dnds_valid = StreamingChainedDataset(
                self.cl_valid,
                valid_datasets,
                finetune_conf.dataset.tpb,
                finetune_conf.dataset.tpb,
                device
            )
        else:
            self.dnds_train, self.dnds_valid = self.__get_finetune_dataset(
                finetune_conf.dataset.to_use[0],
                finetune_conf.dataset,
                self.cl_train,
                self.cl_valid,
                device
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.cl_train.pad_idx)
        pretrained_model_fn = None
        if finetune_conf.pretrain_model_fn is not None:
            pretrained_model_fn = os.path.expandvars(finetune_conf.pretrain_model_fn)
        self.load_model(len(self.vocab), self.criterion, pretrained_model_fn)
        self.load_optimizer(finetune_conf.optimizer)
        self.load_lr_scheduler()
        self.load_checkpoint()

        if finetune_conf.freeze == FreezeOptions.EMB:
            print('Freezing Embeddings')
            self.model.emb.weight.require_grad = False
        elif finetune_conf.freeze == FreezeOptions.EMB_ENC:
            print('Freezing Embeddings + Encoder')
            self.model.emb.weight.require_grad = False
            for param in self.model.enc.parameters():
                param.requires_grad = False
        elif finetune_conf.freeze == FreezeOptions.EMB_ENC_DEC:
            print('Freezing Embeddings + Encoder + Decoder')
            self.model.emb.weight.require_grad = False
            for param in self.model.enc.parameters():
                param.requires_grad = False
            for param in self.model.dec.parameters():
                param.requires_grad = False
        else:
            print('Nothing is freezed')

        if finetune_conf.best_criteria == BestCriteria.GLEU:
            self.use_gleu = True
            src_path = os.path.expandvars(gleu_conf.src)
            refs_path = [os.path.expandvars(r) for r in gleu_conf.refs]
            self.gleu_calculator = GLEU(gleu_conf.n)
            self.gleu_calculator.load_sources(src_path)
            self.gleu_calculator.load_references(refs_path)
            if gleu_conf.preprocess:
                from stylecorrection.utils.preprocess import SpacyBPEPreprocess
                spacy_bpe_pp = SpacyBPEPreprocess.from_conf(preprocess_conf)
                with open(src_path, 'r') as src_f:
                    self.gleu_src_sentences = [spacy_bpe_pp(line.strip()) for line in src_f]
            else:
                with open(src_path, 'r') as src_f:
                    self.gleu_src_sentences = [line.strip() for line in src_f]
        else:
            self.use_gleu = False

    def __infer_src_sentences(self):
        infered_sentences = []
        for sent in self.gleu_src_sentences:
            infered = InferenceOperation.infer(
                self.cl_valid,
                sent,
                self.model,
                self.inference_conf.beam_width,
                self.inference_conf.max_len_scale,
                self.inference_conf.noising_beta,
                self.inference_conf.temperature,
                self.device
            )

            if self.inference_conf.remove_bpe_placeholder:
                infered = infered.replace("@@", "")

            infered_sentences.append(infered.split())
        return infered_sentences

    def run(self):
        train_losses = []
        max_reached = False
        for i in range(self.finetune_conf.training_max.amount):
            self._model.train()
            if self.cl_train.group_indexing:
                current_groups_offsets = self.cl_train.group_offsets.clone()
            for tbi, (t_noised_batch,
                      t_input_key_mask,
                      t_bos_trunc,
                      t_eos_trunc,
                      t_output_key_mask,
                      t_offsets) in enumerate(self.dnds_train):
                if self.finetune_conf.training_max.use == TrainingMax.STEPS and\
                        self.current_training_step >= self.finetune_conf.training_max.amount:
                    max_reached = True
                    print('Max steps reached.')
                if tbi % self.eval_conf.interval == 0 or max_reached:
                    self._model.eval()
                    valid_losses = []
                    with torch.no_grad():
                        for vbi, (v_noised_batch,
                                  v_input_key_mask,
                                  v_bos_trunc,
                                  v_eos_trunc,
                                  v_output_key_mask,
                                  v_offsets) in enumerate(self.dnds_valid):
                            if self.in_multigpu_mode:
                                loss = self._model(
                                    v_bos_trunc,
                                    v_noised_batch,
                                    v_eos_trunc,
                                    v_input_key_mask,
                                    v_output_key_mask,
                                    None
                                )
                                loss = loss.mean()
                            else:
                                out = self._model(
                                    v_noised_batch,
                                    v_eos_trunc,
                                    v_input_key_mask,
                                    v_output_key_mask,
                                    None
                                )

                                loss = self.criterion(
                                    out.contiguous().view(-1, len(self.vocab)),
                                    v_bos_trunc.view(-1)
                                )

                            valid_losses.append(loss.item())
                            if vbi == self.eval_conf.num_valid_batch:
                                break

                        (v_noised_batch,
                         v_input_key_mask,
                         v_bos_trunc,
                         v_eos_trunc,
                         v_output_key_mask,
                         v_offsets) = next(iter(self.dnds_valid))

                        out = self.model(
                            v_noised_batch[:1],
                            v_eos_trunc[:1],
                            v_input_key_mask[:1],
                            v_output_key_mask[:1],
                            None
                        )

                        if out.numel() == 0:
                            print(v_noised_batch)
                        else:
                            enc_input = self.cl_valid.decode_tensor(v_noised_batch[:1])
                            expected_output = self.cl_valid.decode_tensor(v_bos_trunc[:1])
                            predicted_output = self.cl_valid.decode_tensor(out.argmax(dim=2))

                            print()
                            print('Noised sequence : {}'.format(enc_input))
                            print('Expected output : {}'.format(expected_output))
                            print('Predicted output: {}'.format(predicted_output))
                            print()

                        if self.use_gleu:
                            infered_src_sentences = self.__infer_src_sentences()
                            scores = [g for g in self.gleu_calculator.run_iterations(
                                num_iterations=self.gleu_conf.iter,
                                hypothesis=infered_src_sentences,
                                per_sent=self.gleu_conf.sent)]
                            gleu_score = float(scores[0][0])

                        train_loss_mean = torch.tensor(train_losses).mean()
                        valid_loss_mean = torch.tensor(valid_losses).mean()
                        if self.use_gleu:
                            print(
                                '{}: Sentences Processed: {}/{} / Steps {} : Train:{:.4f}, Valid:{:.4f}, GLEU:{:.4f}({:.4f})'.format(
                                    i,
                                    self.cl_train.current_iterating_idx - t_noised_batch.shape[0],
                                    len(self.cl_train),
                                    self.current_training_step,
                                    train_loss_mean,
                                    valid_loss_mean,
                                    gleu_score,
                                    self.best_valid_loss
                                )
                            )
                        else:
                            print(
                                '{}: Sentences Processed: {}/{} / Steps {} : Train:{:.4f}, Valid:{:.4f}({:.4f})'.format(
                                    i,
                                    self.cl_train.current_iterating_idx - t_noised_batch.shape[0],
                                    len(self.cl_train),
                                    self.current_training_step,
                                    train_loss_mean,
                                    valid_loss_mean,
                                    self.best_valid_loss
                                )
                            )

                        if self.use_gleu:
                            if gleu_score != float('inf'):
                                valid_loss = gleu_score * -1
                        else:
                            valid_loss = valid_loss_mean

                        if valid_loss < self.best_valid_loss:
                            self.best_valid_loss = valid_loss
                            self.save_checkpoint(
                                Checkpoints.BEST,
                                current_groups_offsets=current_groups_offsets,
                                current_iterating_idx_offset=self.cl_train.current_iterating_idx - t_noised_batch.shape[0]
                            )

                        self.save_checkpoint(
                            Checkpoints.CURRENT,
                            current_groups_offsets=current_groups_offsets,
                            current_iterating_idx_offset=self.cl_train.current_iterating_idx - t_noised_batch.shape[0]
                        )

                    train_losses.clear()
                    self._model.train()

                if not max_reached:
                    self.optimizer.zero_grad()
                    if self.in_multigpu_mode:
                        loss = self._model(
                            t_bos_trunc,
                            t_noised_batch,
                            t_eos_trunc,
                            t_input_key_mask,
                            t_output_key_mask,
                            None
                        )
                        loss = loss.mean()
                    else:
                        out = self._model(
                            t_noised_batch,
                            t_eos_trunc,
                            t_input_key_mask,
                            t_output_key_mask,
                            None
                        )
                        loss = self.criterion(out.contiguous().view(-1, len(self.vocab)), t_bos_trunc.view(-1))
                    loss.backward()
                    if self.optimizer_conf.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.optimizer_conf.grad_clip_norm)
                    train_losses.append(loss.item())
                    self.optimizer.step()
                    self.current_training_step += 1
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    break
