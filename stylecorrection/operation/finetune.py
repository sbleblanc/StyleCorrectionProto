import os
import h5py
from stylecorrection.operation.common import Operation, Optimizers, LRSchedulers
from stylecorrection.operation.inference import InferenceOperation
from stylecorrection.loaders.corpus import *
from stylecorrection.utils.GLEU import GLEU


class FinetuneStreamingOperation(Operation):

    @classmethod
    def __get_finetune_dataset(cls,
                               which: str,
                               config: dict,
                               train_cl: StreamingH5CorpusLoader,
                               valid_cl: StreamingH5CorpusLoader,
                               device: str):
        if which == 'ca':
            if config['finetune']['dataset']['ca']['shuffler'] == 'chunk':
                shuffler = SentenceShuffler.chunk_shuffler(config['finetune']['dataset']['ca']['min_chunk_ratio'],
                                                           config['finetune']['dataset']['ca']['max_chunk_ratio'])
            elif config['finetune']['dataset']['ca']['shuffler'] == 'normal':
                shuffler = SentenceShuffler.normal_shuffler(config['finetune']['dataset']['ca']['sigma'])

            dnds_train = StreamingCANoiseDataset(train_cl,
                                                 replace_prob=config['finetune']['dataset']['ca']['replace_prob'],
                                                 del_prob=config['finetune']['dataset']['ca']['del_prob'],
                                                 ins_prob=config['finetune']['dataset']['ca']['ins_prob'],
                                                 keep_prob=config['finetune']['dataset']['ca']['keep_prob'],
                                                 mask_prob=config['finetune']['dataset']['ca']['mask_prob'],
                                                 shuffle_prob=config['finetune']['dataset']['ca']['shuffle_prob'],
                                                 shuffler=shuffler,
                                                 tokens_per_batch=config['finetune']['dataset']['tpb'],
                                                 max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                                 device=device)

            dnds_valid = StreamingCANoiseDataset(valid_cl,
                                                 replace_prob=config['finetune']['dataset']['ca']['replace_prob'],
                                                 del_prob=config['finetune']['dataset']['ca']['del_prob'],
                                                 ins_prob=config['finetune']['dataset']['ca']['ins_prob'],
                                                 keep_prob=config['finetune']['dataset']['ca']['keep_prob'],
                                                 mask_prob=config['finetune']['dataset']['ca']['mask_prob'],
                                                 shuffle_prob=config['finetune']['dataset']['ca']['shuffle_prob'],
                                                 shuffler=shuffler,
                                                 tokens_per_batch=config['finetune']['dataset']['tpb'],
                                                 max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                                 device=device)

        elif which == 'parallel':
            dnds_train = StreamingParallelDataset(train_cl,
                                                  split_token=config['finetune']['dataset']['parallel']['split_token'],
                                                  reverse=config['finetune']['dataset']['parallel']['reverse'],
                                                  tokens_per_batch=config['finetune']['dataset']['tpb'],
                                                  max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                                  device=device)
            dnds_valid = StreamingParallelDataset(valid_cl,
                                                  split_token=config['finetune']['dataset']['parallel']['split_token'],
                                                  reverse=config['finetune']['dataset']['parallel']['reverse'],
                                                  tokens_per_batch=config['finetune']['dataset']['tpb'],
                                                  max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                                  device=device)

        return dnds_train, dnds_valid

    def __init__(self,
                 config: dict,
                 device: str):

        super(FinetuneStreamingOperation, self).__init__(config, device)

        self.best_valid_loss = float('inf')
        self.current_training_step = 0

        h5_fn_finetune = os.path.expandvars(config['finetune']['hd5']['finetune']['h5_fn'])
        vocab_h5_fn = os.path.expandvars(config['finetune']['hd5']['vocab']['h5_fn'])
        with h5py.File(vocab_h5_fn, 'r') as h5_file:
            self.vocab = h5_file['vocab'][:]
            if 'additional_special_tokens' in h5_file['vocab'].attrs:
                additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
                vocab_special_chars = self.vocab[5:5 + additional_special_tokens].tolist()
            else:
                vocab_special_chars = []

        self.cl_direct_noise_train, self.cl_direct_noise_valid = StreamingH5CorpusLoader.load_and_split(
            h5_fn_finetune,
            use_split_id=config['finetune']['hd5']['finetune']['valid_split_id'],
            forced_vocab=(self.vocab, vocab_special_chars),
            smoothing_alpha=config['finetune']['hd5']['finetune']['smoothing_alpha'],
            max_sent_len=config['finetune']['max_sent_len'],
            group_by_len=config['finetune']['group_by_len']
        )

        if '+' in config['finetune']['dataset']['to_use']:
            ds_split = config['finetune']['dataset']['to_use'].split('+')
            train_datasets = []
            valid_datasets = []
            for ds in ds_split:
                tds, vds = self.__get_finetune_dataset(
                    ds,
                    config,
                    self.cl_direct_noise_train,
                    self.cl_direct_noise_valid
                )
                train_datasets.append(tds)
                valid_datasets.append(vds)

            self.dnds_train = StreamingChainedDataset(self.cl_direct_noise_train,
                                                      train_datasets,
                                                      config['finetune']['dataset']['tpb'],
                                                      config['finetune']['dataset']['tpb'],
                                                      device)

            self.dnds_valid = StreamingChainedDataset(self.cl_direct_noise_valid,
                                                      valid_datasets, config['finetune']['dataset']['tpb'],
                                                      config['finetune']['dataset']['tpb'],
                                                      device)
        else:
            self.dnds_train, self.dnds_valid = self.__get_finetune_dataset(
                config['finetune']['dataset']['to_use'],
                config, self.cl_direct_noise_train,
                self.cl_direct_noise_valid,
                device
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.cl_direct_noise_train.pad_idx)
        pretrained_model_fn = None
        if config['finetune']['pretrain_model_fn'] is not None:
            pretrained_model_fn = os.path.expandvars(config['finetune']['pretrain_model_fn'])
        self.load_model(len(self.vocab), self.criterion, pretrained_model_fn)
        if config['finetune']['optimizer'] == 'adam':
            self.load_optimizer(Optimizers.ADAM)
        elif config['finetune']['optimizer'] == 'sgd':
            self.load_optimizer(Optimizers.SGD)

        if config['optimizer']['scheduler']['use'] == 'one_cycle':
            self.load_lr_scheduler(LRSchedulers.ONE_CYCLE)

        if config['finetune']['resume_from'] == 'best':
            model_save_fn = os.path.expandvars(config['finetune']['best_model_save_fn'])
        else:
            model_save_fn = os.path.expandvars(config['finetune']['current_model_save_fn'])

        if os.path.exists(model_save_fn):
            with open(model_save_fn, 'rb') as data_file:
                print('Loading from {}'.format(model_save_fn))
                loaded_data = torch.load(data_file, map_location='cpu')
                self.cl_direct_noise_train.current_iterating_idx = loaded_data['current_iterating_idx']
                self.cl_direct_noise_train.current_iterating_order = loaded_data['current_iterating_order']
                self.cl_direct_noise_train.generate_iterating_order = False
                self.optimizer.load_state_dict(loaded_data['optim_state_dict'])
                self.model.load_state_dict(loaded_data['model_state_dict'])
                if "best_valid_loss" in loaded_data:
                    self.best_valid_loss = loaded_data['best_valid_loss']
                if 'current_training_step' in loaded_data:
                    self.current_training_step = loaded_data['current_training_step']
                if config['optimizer']['scheduler']['use'] == 'one_cycle':
                    self.scheduler.load_state_dict(loaded_data['scheduler_state_dict'])

        if config['finetune']['freeze'] == 'emb':
            print('Freezing Embeddings')
            self.model.emb.weight.require_grad = False
        elif config['finetune']['freeze'] == 'enc':
            print('Freezing Embeddings + Encoder')
            self.model.emb.weight.require_grad = False
            for param in self.model.enc.parameters():
                param.requires_grad = False
        elif config['finetune']['freeze'] == 'enc_dec':
            print('Freezing Embeddings + Encoder + Decoder')
            self.model.emb.weight.require_grad = False
            for param in self.model.enc.parameters():
                param.requires_grad = False
            for param in self.model.dec.parameters():
                param.requires_grad = False
        else:
            print('Nothing is freezed')

        if config['finetune']['best_criteria'] == 'gleu':
            self.use_gleu = True
            src_path = os.path.expandvars(self.config['gleu']['src'])
            refs_path = [os.path.expandvars(r) for r in config['gleu']['refs']]
            self.gleu_calculator = GLEU(config['gleu']['n'])
            self.gleu_calculator.load_sources(src_path)
            self.gleu_calculator.load_references(refs_path)
            if config['gleu']['preprocess']:
                from stylecorrection.utils.preprocess import SpacyBPEPreprocess
                codes_fn = os.path.expandvars(config['preprocess']['bpe_codes_fn'])
                bpe_vocab_fn = os.path.expandvars(config['preprocess']['bpe_vocab_fn'])
                spacy_bpe_pp = SpacyBPEPreprocess(codes_fn, bpe_vocab_fn)
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
            infered = InferenceOperation.infer(self.cl_direct_noise_valid,
                                               sent,
                                               self.model,
                                               self.config['inference']['beam_width'],
                                               self.config['inference']['max_len_scale'],
                                               self.config['inference']['noising_beta'],
                                               self.config['inference']['temperature'],
                                               self.device)

            if self.config['inference']['remove_bpe_placeholder']:
                infered = infered.replace("@@", "")

            infered_sentences.append(infered.split())
        return infered_sentences

    def run(self):
        train_losses = []
        max_reached = False
        for i in range(self.config['finetune']['training_max']['amount']):
            self._model.train()
            for tbi, (t_noised_batch,
                      t_input_key_mask,
                      t_bos_trunc,
                      t_eos_trunc,
                      t_output_key_mask,
                      t_offsets) in enumerate(self.dnds_train):
                if self.config['finetune']['training_max']['use'] == 'steps' and\
                        self.current_training_step + 1 >= self.config['finetune']['training_max']['amount']:
                    max_reached = True
                    print('Max steps reached.')
                if tbi % self.config['eval']['interval'] == 0 or max_reached:
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
                                loss = self._model(v_bos_trunc,
                                                   v_noised_batch,
                                                   v_eos_trunc,
                                                   v_input_key_mask,
                                                   v_output_key_mask,
                                                   None)
                                loss = loss.mean()
                            else:
                                out = self._model(v_noised_batch,
                                                  v_eos_trunc,
                                                  v_input_key_mask,
                                                  v_output_key_mask,
                                                  None)

                                loss = self.criterion(out.contiguous().view(-1, len(self.vocab)),
                                                      v_bos_trunc.view(-1))

                            valid_losses.append(loss.item())
                            if vbi == self.config['eval']['num_valid_batch']:
                                break

                        (v_noised_batch,
                         v_input_key_mask,
                         v_bos_trunc,
                         v_eos_trunc,
                         v_output_key_mask,
                         v_offsets) = next(iter(self.dnds_valid))

                        out = self.model(v_noised_batch[:1],
                                         v_eos_trunc[:1],
                                         v_input_key_mask[:1],
                                         v_output_key_mask[:1],
                                         None)

                        if out.numel() == 0:
                            print(v_noised_batch)
                        else:
                            enc_input = self.cl_direct_noise_valid.decode_tensor(v_noised_batch[:1])
                            expected_output = self.cl_direct_noise_valid.decode_tensor(v_bos_trunc[:1])
                            predicted_output = self.cl_direct_noise_valid.decode_tensor(out.argmax(dim=2))

                            print()
                            print('Noised sequence : {}'.format(enc_input))
                            print('Expected output : {}'.format(expected_output))
                            print('Predicted output: {}'.format(predicted_output))
                            print()

                        if self.use_gleu:
                            infered_src_sentences = self.__infer_src_sentences()
                            scores = [g for g in self.gleu_calculator.run_iterations(
                                num_iterations=self.config['gleu']['iter'],
                                hypothesis=infered_src_sentences,
                                per_sent=self.config['gleu']['sent'])]
                            gleu_score = float(scores[0][0])

                        train_loss_mean = torch.tensor(train_losses).mean()
                        valid_loss_mean = torch.tensor(valid_losses).mean()
                        if self.use_gleu:
                            print('{}: Sentences Processed: {}/{} / Steps {} : Train:{:.4f}, Valid:{:.4f}, GLEU:{:.4f}({:.4f})'.format(
                                    i,
                                    self.cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                    len(self.cl_direct_noise_train),
                                    self.current_training_step,
                                    train_loss_mean,
                                    valid_loss_mean,
                                    gleu_score,
                                    self.best_valid_loss))
                        else:
                            print('{}: Sentences Processed: {}/{} / Steps {} : Train:{:.4f}, Valid:{:.4f}({:.4f})'.format(
                                i,
                                self.cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                len(self.cl_direct_noise_train),
                                self.current_training_step,
                                train_loss_mean,
                                valid_loss_mean,
                                self.best_valid_loss))

                        if self.use_gleu:
                            if gleu_score != float('inf'):
                                valid_loss = gleu_score * -1
                        else:
                            valid_loss = valid_loss_mean

                        if valid_loss < self.best_valid_loss:
                            save_fn = os.path.expandvars(self.config['finetune']['best_model_save_fn'])
                            with open(save_fn, 'wb') as out_file:
                                to_save = {
                                    'current_iterating_idx': self.cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                    'current_iterating_order': self.cl_direct_noise_train.current_iterating_order,
                                    'model_state_dict': self.model.state_dict(),
                                    'optim_state_dict': self.optimizer.state_dict(),
                                    'best_valid_loss': valid_loss,
                                    'current_training_step': self.current_training_step}
                                if self.scheduler:
                                    to_save['scheduler_state_dict'] = self.scheduler.state_dict()
                                torch.save(to_save, out_file)
                            self.best_valid_loss = valid_loss

                        save_fn = os.path.expandvars(self.config['finetune']['current_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            to_save = {
                                'current_iterating_idx': self.cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                'current_iterating_order': self.cl_direct_noise_train.current_iterating_order,
                                'model_state_dict': self.model.state_dict(),
                                'optim_state_dict': self.optimizer.state_dict(),
                                'best_valid_loss': self.best_valid_loss,
                                'current_training_step': self.current_training_step}
                            if self.scheduler:
                                to_save['scheduler_state_dict'] = self.scheduler.state_dict()
                            torch.save(to_save, out_file)

                    train_losses.clear()
                    self._model.train()

                if not max_reached:
                    self.optimizer.zero_grad()
                    if self.in_multigpu_mode:
                        loss = self._model(t_bos_trunc,
                                           t_noised_batch,
                                           t_eos_trunc,
                                           t_input_key_mask,
                                           t_output_key_mask,
                                           None)
                        loss = loss.mean()
                    else:
                        out = self._model(t_noised_batch,
                                          t_eos_trunc,
                                          t_input_key_mask,
                                          t_output_key_mask,
                                          None)
                        loss = self.criterion(out.contiguous().view(-1, len(self.vocab)), t_bos_trunc.view(-1))
                    loss.backward()
                    if self.config['optimizer']['grad_clip_norm'] > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config['optimizer']['grad_clip_norm'])
                    train_losses.append(loss.item())
                    self.optimizer.step()
                    self.current_training_step += 1
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    break
