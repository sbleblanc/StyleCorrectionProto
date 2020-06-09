import argparse
import yaml
import os
import torch.optim as optim
from stylecorrection.loaders.corpus import *
from stylecorrection.models.wrappers import *
from stylecorrection.models.transformer import TransformerS2S

def get_finetune_dataset(which, config, train_cl, valid_cl):
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
        dnds_train = StreamingParallelDataset(cl_direct_noise_train,
                                              split_token=config['finetune']['dataset']['parallel']['split_token'],
                                              reverse=config['finetune']['dataset']['parallel']['reverse'],
                                              tokens_per_batch=config['finetune']['dataset']['tpb'],
                                              max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                              device=device)
        dnds_valid = StreamingParallelDataset(cl_direct_noise_valid,
                                              split_token=config['finetune']['dataset']['parallel']['split_token'],
                                              reverse=config['finetune']['dataset']['parallel']['reverse'],
                                              tokens_per_batch=config['finetune']['dataset']['tpb'],
                                              max_trainable_tokens=config['finetune']['dataset']['tpb'],
                                              device=device)

    return dnds_train, dnds_valid


device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
params = parser.parse_args()

with open(params.config, 'r') as in_file:
    config = yaml.load(in_file, Loader=yaml.FullLoader)

if config['mode'] == 'eval':
    print('Starting manual evaluation...')
    if config['manual_eval']['force_cpu']:
        device = 'cpu'

    vocab_path = os.path.expandvars(config['manual_eval']['h5']['vocab'])
    with h5py.File(vocab_path, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
        else:
            vocab_special_chars = []

    ft_corpus_path = os.path.expandvars(config['manual_eval']['h5']['ft_corpus'])
    cl = StreamingH5CorpusLoader.load_and_split(
        ft_corpus_path,
        use_split_id=config['manual_eval']['h5']['ft_corpus_split'],
        forced_vocab=(vocab, vocab_special_chars)
    )[0]

    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )

    pretrained_mdl_path = os.path.expandvars(config['manual_eval']['pretrained_model'])
    with open(pretrained_mdl_path, 'rb') as in_file:
        loaded_data = torch.load(in_file, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'])
    model.to(device)
    model.eval()

    for ds, cs in zip(config['manual_eval']['sample_corrections']['dirty'],
                      config['manual_eval']['sample_corrections']['clean']):
        test_sentence = cl.encode_sentence(ds).to(device)

        with torch.no_grad():
            res = model.beam_decode_2(
                test_sentence,
                torch.tensor([cl.bos_idx], dtype=torch.long),
                beam_width=config['manual_eval']['beam_width'],
                max_len=int(len(test_sentence) * 1.5),
                end_token=cl.eos_idx,
                return_scores=True,
                device=device
            )

        print("IN: {}".format(ds))
        print("EXPECTED: {}".format(cs))
        for s, b in res:
            decoded = cl.decode_tensor(b)
            print('\t({:.4f}) : {}'.format(s, decoded[0]))

if config['mode'] == 'hd5_gen':
    print('Creating hd5 dataset...')
    additional_tokens = []
    if config['hd5_gen']['additional_tokens'] is not None:
        additional_tokens = config['hd5_gen']['additional_tokens']
    h5_fn = os.path.expandvars(config['hd5_gen']['h5_fn'])
    StreamingH5CorpusLoader.create_from_compressed(
        h5_fn,
        os.path.expandvars(config['hd5_gen']['corpus_tar_gz']),
        lambda x: x.strip().split(' '),
        lambda x: x.lower(),
        config['hd5_gen']['topk'],
        config['hd5_gen']['max_len'],
        additional_tokens
    )
    print('DONE')

elif config['mode'] == 'gen_split':
    print('Generating train/valid split...')
    h5_fn = os.path.expandvars(config['gen_split']['h5_fn'])
    StreamingH5CorpusLoader.generate_split(
        h5_fn,
        config['gen_split']['valid_ratio']
    )
    print('DONE')

elif config['mode'] == 'vocab_dump':
    print('Starting vocab dumping...')
    corpus_h5_fn = os.path.expandvars(config['vocab_dump']['corpus_h5_fn'])
    vocab_h5_fn = os.path.expandvars(config['vocab_dump']['vocab_h5_fn'])
    cl = StreamingH5CorpusLoader.load_and_split(
        corpus_h5_fn,
        config['vocab_dump']['valid_split_id'],
        config['vocab_dump']['vocab_topk'],
        config['vocab_dump']['min_freq'],
        device=device
    )[0]
    cl.dump_vocab(vocab_h5_fn)
    print('DONE')

elif config['mode'] == 'pretrain':
    print('Starting Pretraining...')
    corpus_h5_fn = os.path.expandvars(config['pretrain']['hd5']['corpus_fn'])
    vocab_h5_fn = os.path.expandvars(config['pretrain']['hd5']['vocab_fn'])
    with h5py.File(vocab_h5_fn, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
    cl = H5CorpusLoader.load_and_split(
        corpus_h5_fn,
        use_split_id=config['pretrain']['hd5']['valid_split_id'],
        forced_vocab=vocab,
        device=device
    )
    if config['pretrain']['algo'] == 'bart':
        print('Using text infilling (BART)')
        pds = BARTPretrainingDataset(cl, device=device)
    else:
        pds = PretrainingDataset(cl, device=device)

    model = TransformerS2S(
        len(cl.vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )
    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if config['pretrain']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['optimizer']['adam']['lr'],
                               betas=(config['optimizer']['adam']['beta_1'], config['optimizer']['adam']['beta_2']),
                               eps=config['optimizer']['adam']['eps'])
    elif config['pretrain']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['optimizer']['sgd']['lr'],
                              momentum=config['optimizer']['sgd']['momentum'],
                              weight_decay=config['optimizer']['sgd']['weight_decay'],
                              nesterov=config['optimizer']['sgd']['nesterov'])
    criterion = nn.CrossEntropyLoss(ignore_index=cl.pad_idx).to(device)

    train_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    bs = config['pretrain']['bs']

    for i in range(config['pretrain']['max_epoch']):
        model.train()
        for tbi, (t_enc_in, t_enc_in_key_mask, t_dec_out, t_dec_in, t_dec_in_key_mask, t_offsets) in enumerate(pds(bs=bs)):

            if tbi % config['eval']['interval'] == 0:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets) in enumerate(pds(bs=bs, which='valid')):
                        out = model(v_enc_in, v_dec_in, v_enc_in_key_mask, v_dec_in_key_mask, v_offsets)
                        loss = criterion(out.contiguous().view(-1, len(cl.vocab)), v_dec_out.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets = next(pds(bs=1, which='valid'))
                    out = model(v_enc_in, v_dec_in, v_enc_in_key_mask, v_dec_in_key_mask, v_offsets)
                    if out.numel() == 0:
                        print(v_enc_in)
                    else:
                        enc_input = cl.decode_tensor(v_enc_in)
                        expected_output = cl.decode_tensor(v_dec_out)
                        predicted_output = cl.decode_tensor(out.argmax(dim=2))

                        print()
                        print('Masked sequence : {}'.format(enc_input))
                        print('Expected segment : {}'.format(expected_output))
                        print('Predicted segment: {}'.format(predicted_output))
                        print()

                    train_loss_mean = torch.tensor(train_losses).mean()
                    valid_loss_mean = torch.tensor(valid_losses).mean()
                    print('{}: Batch {}/{} : Train:{:.4f}, Valid:{:.4f}'.format(i, tbi*bs, pds.get_num_sentences('train'), train_loss_mean, valid_loss_mean))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['pretrain']['best_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            torch.save(model.state_dict(), out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                    save_fn = os.path.expandvars(config['pretrain']['current_model_save_fn'])
                    with open(save_fn, 'wb') as out_file:
                        torch.save(model.state_dict(), out_file)

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
            loss = criterion(out.contiguous().view(-1, len(cl.vocab)), t_dec_out.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

        if patience_counter > config['pretrain']['valid_patience']:
            print('Patience threshold reached')
            break
    print('DONE')

elif config['mode'] == 'finetune':
    h5_fn_finetune = os.path.expandvars(config['finetune']['hd5']['finetune']['h5_fn'])
    vocab_h5_fn = os.path.expandvars(os.path.expandvars(config['finetune']['hd5']['vocab']['h5_fn']))
    with h5py.File(vocab_h5_fn, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
    cl_direct_noise = H5CorpusLoader.load_and_split(
        h5_fn_finetune,
        use_split_id=config['finetune']['hd5']['finetune']['valid_split_id'],
        forced_vocab=vocab,
        smoothing_alpha=config['finetune']['hd5']['finetune']['smoothing_alpha']
    )
    if config['finetune']['dataset']['to_use'] == 'CA':
        dnds = CANoiseDataset(cl_direct_noise,
                              replace_prob=config['finetune']['dataset']['ca']['replace_prob'],
                              del_prob=config['finetune']['dataset']['ca']['del_prob'],
                              ins_prob=config['finetune']['dataset']['ca']['ins_prob'],
                              keep_prob=config['finetune']['dataset']['ca']['keep_prob'],
                              mask_prob=config['finetune']['dataset']['ca']['mask_prob'],
                              sigma=config['finetune']['dataset']['ca']['sigma'],
                              device=device)
    else:
        dnds = DirectNoiseDataset(cl_direct_noise,
                                  del_prob=config['finetune']['dataset']['dn']['del_prob'],
                                  ins_prob=config['finetune']['dataset']['dn']['ins_prob'],
                                  keep_prob=config['finetune']['dataset']['dn']['keep_prob'],
                                  mask_prob=config['finetune']['dataset']['dn']['mask_prob'],
                                  device=device)

    # sample_enc_inputs_lst = []
    # longest = 0
    # for sentence in config['sample_corrections']['dirty']:
    #     sample_enc_inputs_lst.append(cl_pretrain.encode_sentence(sentence))
    #     if sample_enc_inputs_lst[-1].shape[0] > longest:
    #         longest = sample_enc_inputs_lst[-1].shape[0]
    # sample_enc_input = torch.empty([len(sample_enc_inputs_lst), longest], dtype=torch.long).fill_(cl_pretrain.mask_idx).to(device)
    # sample_enc_mask = torch.zeros([len(sample_enc_inputs_lst), longest]).bool().to(device)
    # for i, sample in enumerate(sample_enc_inputs_lst):
    #     sample_enc_input[i, :sample.shape[0]] = sample
    #     sample_enc_mask[i, sample.shape[0]:] = True
    # sample_expected_outputs = config['sample_corrections']['clean']


    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )
    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    model_fn = os.path.expandvars(config['finetune']['pretrain_model_fn'])
    with open(model_fn, 'rb') as in_file:
        model.load_state_dict(torch.load(in_file, map_location=device))

    model.to(device)

    if config['finetune']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['optimizer']['adam']['lr'],
                               betas=(config['optimizer']['adam']['beta_1'], config['optimizer']['adam']['beta_2']),
                               eps=config['optimizer']['adam']['eps'])
    elif config['finetune']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['optimizer']['sgd']['lr'],
                              momentum=config['optimizer']['sgd']['momentum'],
                              weight_decay=config['optimizer']['sgd']['weight_decay'],
                              nesterov=config['optimizer']['sgd']['nesterov'])

    criterion = nn.CrossEntropyLoss(ignore_index=cl_direct_noise.pad_idx).to(device)
    train_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    bs = config['finetune']['bs']
    for i in range(config['finetune']['max_epoch']):
        model.train()
        for tbi, (t_noised_batch, t_input_key_mask, t_eos_trunc, t_bos_trunc, t_output_key_mask) in enumerate(dnds(bs=bs)):

            if tbi % config['eval']['interval'] == 0:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (
                    v_noised_batch, v_input_key_mask, v_eos_trunc, v_bos_trunc, v_output_key_mask) in enumerate(
                            dnds(bs=bs, which='valid')):
                        out = model(v_noised_batch, v_eos_trunc, v_input_key_mask, v_output_key_mask, None)
                        loss = criterion(out.contiguous().view(-1, len(cl_direct_noise.vocab)), v_bos_trunc.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_noised_batch, v_input_key_mask, v_eos_trunc, v_bos_trunc, v_output_key_mask = next(
                        dnds(bs=1, which='valid'))
                    out = model(v_noised_batch, v_eos_trunc, v_input_key_mask, v_output_key_mask, None)
                    if out.numel() == 0:
                        print(v_noised_batch)
                    else:
                        enc_input = cl_direct_noise.decode_tensor(v_noised_batch)
                        expected_output = cl_direct_noise.decode_tensor(v_bos_trunc)
                        predicted_output = cl_direct_noise.decode_tensor(out.argmax(dim=2))

                        print()
                        print('Noised sequence : {}'.format(enc_input))
                        print('Expected output : {}'.format(expected_output))
                        print('Predicted output: {}'.format(predicted_output))
                        print()

                    train_loss_mean = torch.tensor(train_losses).mean()
                    valid_loss_mean = torch.tensor(valid_losses).mean()
                    print('{}: Batch {}/{} : Train:{:.4f}, Valid:{:.4f}'.format(i, tbi * bs,
                                                                                dnds.get_num_sentences('train'),
                                                                                train_loss_mean, valid_loss_mean))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['finetune']['model_save_fn'])
                        with open(save_fn, 'wb') as out_file:

                            torch.save(model.state_dict(), out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(t_noised_batch, t_eos_trunc, t_input_key_mask, t_output_key_mask, None)
            loss = criterion(out.contiguous().view(-1, len(cl_direct_noise.vocab)), t_bos_trunc.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

elif config['mode'] == 'pretrain_streaming':
    print('Starting Pretraining...')
    corpus_h5_fn = os.path.expandvars(config['pretrain']['hd5']['corpus_fn'])
    vocab_h5_fn = os.path.expandvars(config['pretrain']['hd5']['vocab_fn'])
    with h5py.File(vocab_h5_fn, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5+additional_special_tokens].tolist()
        else:
            vocab_special_chars = []
    cl_train, cl_valid = StreamingH5CorpusLoader.load_and_split(
        corpus_h5_fn,
        use_split_id=config['pretrain']['hd5']['valid_split_id'],
        forced_vocab=(vocab, vocab_special_chars),
        max_sent_len=config['pretrain']['max_sent_len'],
        group_by_len=config['pretrain']['hd5']['group_by_len'],
        device=device
    )
    if config['pretrain']['algo'] == 'bart':
        print('Using text infilling (BART)')
        pds_train = StreamingBARTPretrainingDataset(cl_train, tokens_per_batch=config['pretrain']['tpb'], max_trainable_tokens=config['pretrain']['mttpb'], device=device)
        pds_valid = StreamingBARTPretrainingDataset(cl_valid, tokens_per_batch=config['pretrain']['tpb'], max_trainable_tokens=config['pretrain']['mttpb'], device=device)
    else:
        pds_train = StreamingMASSPretrainingDataset(cl_train, tokens_per_batch=config['pretrain']['tpb'], max_trainable_tokens=config['pretrain']['mttpb'], device=device)
        pds_valid = StreamingMASSPretrainingDataset(cl_valid, tokens_per_batch=config['pretrain']['tpb'], max_trainable_tokens=config['pretrain']['mttpb'], device=device)

    model = TransformerS2S(
        len(cl_train.vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers'],
        config['TransformerS2S']['activation']
    )

    criterion = nn.CrossEntropyLoss(ignore_index=cl_train.pad_idx)

    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        in_multigpu_mode = True
        model = DataParallelCELWrapper(model, criterion, len(cl_train.vocab))
        model = nn.DataParallel(model)
    else:
        in_multigpu_mode = False
    model.to(device)

    if config['pretrain']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['optimizer']['adam']['lr'],
                               betas=(config['optimizer']['adam']['beta_1'], config['optimizer']['adam']['beta_2']),
                               eps=config['optimizer']['adam']['eps'],
                               weight_decay=config['optimizer']['adam']['weight_decay'])
    elif config['pretrain']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['optimizer']['sgd']['lr'],
                              momentum=config['optimizer']['sgd']['momentum'],
                              weight_decay=config['optimizer']['sgd']['weight_decay'],
                              nesterov=config['optimizer']['sgd']['nesterov'])

    if config['optimizer']['scheduler']['use'] == 'one_cycle':
        pct = config['optimizer']['scheduler']['one_cycle']['warmup_steps'] / \
              config['optimizer']['scheduler']['one_cycle']['total_steps']
        print('Scheduler Pct: {:%}'.format(pct))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config['optimizer']['scheduler']['one_cycle']['max_lr'],
            div_factor=config['optimizer']['scheduler']['one_cycle']['initial_lr_div'],
            final_div_factor=config['optimizer']['scheduler']['one_cycle']['final_lr_div'],
            total_steps=config['optimizer']['scheduler']['one_cycle']['total_steps'],
            anneal_strategy=config['optimizer']['scheduler']['one_cycle']['anneal_strategy'],
            pct_start=pct,
            last_epoch=-1,
            cycle_momentum=False
        )

    if config['pretrain']['resume_from'] == 'best':
        model_save_fn = os.path.expandvars(config['pretrain']['best_model_save_fn'])
    else:
        model_save_fn = os.path.expandvars(config['pretrain']['current_model_save_fn'])

    best_valid_loss = float('inf')
    current_training_step = 0
    if os.path.exists(model_save_fn):
        with open(model_save_fn, 'rb') as data_file:
            print('Loading from {}'.format(model_save_fn))
            loaded_data = torch.load(data_file, map_location='cpu')
            if cl_train.group_indexing:
                cl_train.state = {k: loaded_data[k] for k in ['current_group', 'current_group_selector', 'current_group_offsets'] if k in loaded_data}
            else:
                cl_train.state = {k: loaded_data[k] for k in
                                  ['current_iterating_idx', 'current_iterating_order'] if
                                  k in loaded_data}
            cl_train.generate_iterating_order = False
            optimizer.load_state_dict(loaded_data['optim_state_dict'])
            if in_multigpu_mode:
                model.module.model.load_state_dict(loaded_data['model_state_dict'])
            else:
                model.load_state_dict(loaded_data['model_state_dict'])
            if 'best_valid_loss' in loaded_data:
                best_valid_loss = loaded_data['best_valid_loss']
            if 'current_training_step' in loaded_data:
                current_training_step = loaded_data['current_training_step']
            if config['optimizer']['scheduler']['use'] == 'one_cycle':
                scheduler.load_state_dict(loaded_data['scheduler_state_dict'])
                print(scheduler.state_dict())



    train_losses = []
    patience_counter = 0
    max_reached = False
    for i in range(config['pretrain']['training_max']['amount']):
        model.train()
        if cl_train.group_indexing:
            current_groups_offsets = cl_train.group_offsets.clone()
        for tbi, (t_enc_in, t_enc_in_key_mask, t_dec_out, t_dec_in, t_dec_in_key_mask, t_offsets) in enumerate(pds_train):
            if config['pretrain']['training_max']['use'] == 'steps' and current_training_step >= \
                    config['pretrain']['training_max']['amount']:
                print('Max steps reached.')
                max_reached = True
            if tbi % config['eval']['interval'] == 0 or max_reached:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets) in enumerate(pds_valid):
                        if in_multigpu_mode:
                            loss = model(v_dec_out, v_enc_in, v_dec_in, v_enc_in_key_mask, v_dec_in_key_mask, v_offsets)
                            loss = loss.mean()
                        else:
                            out = model(v_enc_in, v_dec_in, v_enc_in_key_mask, v_dec_in_key_mask, v_offsets)
                            loss = criterion(out.contiguous().view(-1, len(cl_valid.vocab)), v_dec_out.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets = next(iter(pds_valid))
                    if in_multigpu_mode:
                        if config['pretrain']['algo'] == 'bart':
                            out = model.module.model(v_enc_in[:1], v_dec_in[:1], v_enc_in_key_mask[:1], v_dec_in_key_mask[:1], None)
                        else:
                            out = model.module.model(v_enc_in[:1], v_dec_in[:1], v_enc_in_key_mask[:1], v_dec_in_key_mask[:1], v_offsets[:1])
                    else:
                        if config['pretrain']['algo'] == 'bart':
                            out = model(v_enc_in[:1], v_dec_in[:1], v_enc_in_key_mask[:1], v_dec_in_key_mask[:1], None)
                        else:
                            out = model(v_enc_in[:1], v_dec_in[:1], v_enc_in_key_mask[:1], v_dec_in_key_mask[:1], v_offsets[:1])
                    if out.numel() == 0:
                        print(v_enc_in)
                    else:
                        enc_input = cl_train.decode_tensor(v_enc_in[:1])
                        expected_output = cl_train.decode_tensor(v_dec_out[:1])
                        predicted_output = cl_train.decode_tensor(out.argmax(dim=2))

                        print()
                        print('Masked sequence : {}'.format(enc_input))
                        print('Expected segment : {}'.format(expected_output))
                        print('Predicted segment: {}'.format(predicted_output))
                        print()

                    train_loss_mean = torch.tensor(train_losses).mean()
                    valid_loss_mean = torch.tensor(valid_losses).mean()
                    print('{}: Sentences Processed {}/{} / Num. Steps {} : Train:{:.4f}, Valid:{:.4f}'.format(i, cl_train.current_iterating_idx - t_enc_in.shape[0], len(cl_train), current_training_step, train_loss_mean, valid_loss_mean))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['pretrain']['best_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            to_save = {
                                'optim_state_dict': optimizer.state_dict(),
                                'best_valid_loss': valid_loss_mean,
                                'current_training_step': current_training_step
                            }
                            to_save.update(cl_train.state)
                            if cl_train.group_indexing:
                                to_save['current_group_offsets'] = current_groups_offsets
                            else:
                                to_save['current_iterating_idx'] -= t_enc_in.shape[0]
                            if in_multigpu_mode:
                                to_save['model_state_dict'] = model.module.model.state_dict()
                            else:
                                to_save['model_state_dict'] = model.state_dict()
                            if config['optimizer']['scheduler']['use'] == 'one_cycle':
                                to_save['scheduler_state_dict'] = scheduler.state_dict()
                            torch.save(to_save, out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                    save_fn = os.path.expandvars(config['pretrain']['current_model_save_fn'])
                    with open(save_fn, 'wb') as out_file:
                        to_save = {
                            'optim_state_dict': optimizer.state_dict(),
                            'best_valid_loss': best_valid_loss,
                            'current_training_step': current_training_step
                        }
                        to_save.update(cl_train.state)
                        if cl_train.group_indexing:
                            to_save['current_group_offsets'] = current_groups_offsets
                        else:
                            to_save['current_iterating_idx'] -= t_enc_in.shape[0]
                        if in_multigpu_mode:
                            to_save['model_state_dict'] = model.module.model.state_dict()
                        else:
                            to_save['model_state_dict'] = model.state_dict()
                        if config['optimizer']['scheduler']['use'] == 'one_cycle':
                            to_save['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(to_save, out_file)

                train_losses.clear()
                model.train()

            if not max_reached:
                optimizer.zero_grad()
                if in_multigpu_mode:
                    loss = model(t_dec_out, t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
                    loss = loss.mean()
                else:
                    out = model(t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
                    loss = criterion(out.contiguous().view(-1, len(cl_train.vocab)), t_dec_out.view(-1))
                loss.backward()
                if config['optimizer']['grad_clip_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip_norm'])
                train_losses.append(loss.item())
                optimizer.step()
                current_training_step += 1
                current_groups_offsets = cl_train.group_offsets.clone()
                if config['optimizer']['scheduler']['use'] == 'one_cycle':
                    scheduler.step()
            else:
                break

    print('DONE')

elif config['mode'] == 'finetune_streaming':
    h5_fn_finetune = os.path.expandvars(config['finetune']['hd5']['finetune']['h5_fn'])
    vocab_h5_fn = os.path.expandvars(os.path.expandvars(config['finetune']['hd5']['vocab']['h5_fn']))
    with h5py.File(vocab_h5_fn, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
        else:
            vocab_special_chars = []
    cl_direct_noise_train, cl_direct_noise_valid = StreamingH5CorpusLoader.load_and_split(
        h5_fn_finetune,
        use_split_id=config['finetune']['hd5']['finetune']['valid_split_id'],
        forced_vocab=(vocab, vocab_special_chars),
        smoothing_alpha=config['finetune']['hd5']['finetune']['smoothing_alpha'],
        max_sent_len=config['finetune']['max_sent_len'],
        group_by_len=config['finetune']['group_by_len']
    )

    if '+' in config['finetune']['dataset']['to_use']:
        ds_split = config['finetune']['dataset']['to_use'].split('+')
        train_datasets = []
        valid_datasets = []
        for ds in ds_split:
            tds, vds = get_finetune_dataset(ds, config, cl_direct_noise_train, cl_direct_noise_valid)
            train_datasets.append(tds)
            valid_datasets.append(vds)
        dnds_train = StreamingChainedDataset(cl_direct_noise_train, train_datasets,
                                             config['finetune']['dataset']['tpb'], config['finetune']['dataset']['tpb'],
                                             device)
        dnds_valid = StreamingChainedDataset(cl_direct_noise_valid, valid_datasets,
                                             config['finetune']['dataset']['tpb'], config['finetune']['dataset']['tpb'],
                                             device)
    else:
        dnds_train, dnds_valid = get_finetune_dataset(config['finetune']['dataset']['to_use'], config, cl_direct_noise_train, cl_direct_noise_valid)

    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers'],
        config['TransformerS2S']['activation']
    )

    criterion = nn.CrossEntropyLoss(ignore_index=cl_direct_noise_train.pad_idx)

    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        in_multigpu_mode = True
        model = DataParallelCELWrapper(model, criterion, len(vocab))
        model = nn.DataParallel(model)
    else:
        in_multigpu_mode = False
    model.to(device)

    if config['finetune']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['optimizer']['adam']['lr'],
                               betas=(config['optimizer']['adam']['beta_1'], config['optimizer']['adam']['beta_2']),
                               eps=config['optimizer']['adam']['eps'],
                               weight_decay=config['optimizer']['adam']['weight_decay'])
    elif config['finetune']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['optimizer']['sgd']['lr'],
                              momentum=config['optimizer']['sgd']['momentum'],
                              weight_decay=config['optimizer']['sgd']['weight_decay'],
                              nesterov=config['optimizer']['sgd']['nesterov'])

    if config['optimizer']['scheduler']['use'] == 'one_cycle':
        pct = config['optimizer']['scheduler']['one_cycle']['warmup_steps'] / \
              config['optimizer']['scheduler']['one_cycle']['total_steps']
        print('Scheduler Pct: {:%}'.format(pct))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config['optimizer']['scheduler']['one_cycle']['max_lr'],
            div_factor=config['optimizer']['scheduler']['one_cycle']['initial_lr_div'],
            final_div_factor=config['optimizer']['scheduler']['one_cycle']['final_lr_div'],
            total_steps=config['optimizer']['scheduler']['one_cycle']['total_steps'],
            anneal_strategy=config['optimizer']['scheduler']['one_cycle']['anneal_strategy'],
            pct_start=pct,
            last_epoch=-1,
            cycle_momentum=False
        )

    if config['finetune']['resume_from'] == 'best':
        model_save_fn = os.path.expandvars(config['finetune']['best_model_save_fn'])
    else:
        model_save_fn = os.path.expandvars(config['finetune']['current_model_save_fn'])

    best_valid_loss = float('inf')
    current_training_step = 0
    if os.path.exists(model_save_fn):
        with open(model_save_fn, 'rb') as data_file:
            print('Loading from {}'.format(model_save_fn))
            loaded_data = torch.load(data_file, map_location='cpu')
            cl_direct_noise_train.current_iterating_idx = loaded_data['current_iterating_idx']
            cl_direct_noise_train.current_iterating_order = loaded_data['current_iterating_order']
            cl_direct_noise_train.generate_iterating_order = False
            optimizer.load_state_dict(loaded_data['optim_state_dict'])
            if in_multigpu_mode:
                model.module.model.load_state_dict(loaded_data['model_state_dict'])
            else:
                model.load_state_dict(loaded_data['model_state_dict'])
            if "best_valid_loss" in loaded_data:
                best_valid_loss = loaded_data['best_valid_loss']
            if 'current_training_step' in loaded_data:
                current_training_step = loaded_data['current_training_step']
            if config['optimizer']['scheduler']['use'] == 'one_cycle':
                scheduler.load_state_dict(loaded_data['scheduler_state_dict'])
                print(scheduler.state_dict())

    if config['finetune']['pretrain_model_fn'] is not None:
        model_fn = os.path.expandvars(config['finetune']['pretrain_model_fn'])
        with open(model_fn, 'rb') as in_file:
            loaded_data = torch.load(in_file, map_location=device)
            if in_multigpu_mode:
                model.module.model.load_state_dict(loaded_data['model_state_dict'])
            else:
                model.load_state_dict(loaded_data['model_state_dict'])
            loaded_data = None

    if config['finetune']['freeze'] == 'emb':
        print('Freezing Embeddings')
        if in_multigpu_mode:
            model.module.model.emb.weight.require_grad = False
        else:
            model.emb.weight.require_grad = False
    elif config['finetune']['freeze'] == 'enc':
        print('Freezing Embeddings + Encoder')
        if in_multigpu_mode:
            model.module.model.emb.weight.require_grad = False
            for param in model.module.model.enc.parameters():
                param.requires_grad = False
        else:
            model.emb.weight.require_grad = False
            for param in model.enc.parameters():
                param.requires_grad = False
    elif config['finetune']['freeze'] == 'enc_dec':
        print('Freezing Embeddings + Encoder + Decoder')
        if in_multigpu_mode:
            model.module.model.emb.weight.require_grad = False
            for param in model.module.model.enc.parameters():
                param.requires_grad = False
            for param in model.module.model.dec.parameters():
                param.requires_grad = False
        else:
            model.emb.weight.require_grad = False
            for param in model.enc.parameters():
                param.requires_grad = False
            for param in model.dec.parameters():
                param.requires_grad = False
    else:
        print('Nothing is freezed')

    train_losses = []
    max_reached = False
    patience_counter = 0
    for i in range(config['finetune']['training_max']['amount']):
        model.train()
        for tbi, (t_noised_batch, t_input_key_mask, t_bos_trunc, t_eos_trunc, t_output_key_mask, t_offsets) in enumerate(dnds_train):
            if config['finetune']['training_max']['use'] == 'steps' and current_training_step + 1 >= \
                    config['finetune']['training_max']['amount']:
                max_reached = True
                print('Max steps reached.')
            if tbi % config['eval']['interval'] == 0 or max_reached:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (v_noised_batch, v_input_key_mask, v_bos_trunc, v_eos_trunc, v_output_key_mask, v_offsets) in enumerate(dnds_valid):
                        if in_multigpu_mode:
                            loss = model(v_bos_trunc, v_noised_batch, v_eos_trunc, v_input_key_mask, v_output_key_mask, None)
                            loss = loss.mean()
                        else:
                            out = model(v_noised_batch, v_eos_trunc, v_input_key_mask, v_output_key_mask, None)
                            loss = criterion(out.contiguous().view(-1, len(vocab)), v_bos_trunc.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_noised_batch, v_input_key_mask, v_bos_trunc, v_eos_trunc, v_output_key_mask, v_offsets = next(iter(dnds_valid))
                    if in_multigpu_mode:
                        out = model.module.model(v_noised_batch[:1], v_eos_trunc[:1], v_input_key_mask[:1], v_output_key_mask[:1], None)
                    else:
                        out = model(v_noised_batch[:1], v_eos_trunc[:1], v_input_key_mask[:1], v_output_key_mask[:1], None)
                    if out.numel() == 0:
                        print(v_noised_batch)
                    else:
                        enc_input = cl_direct_noise_valid.decode_tensor(v_noised_batch[:1])
                        expected_output = cl_direct_noise_valid.decode_tensor(v_bos_trunc[:1])
                        predicted_output = cl_direct_noise_valid.decode_tensor(out.argmax(dim=2))

                        print()
                        print('Noised sequence : {}'.format(enc_input))
                        print('Expected output : {}'.format(expected_output))
                        print('Predicted output: {}'.format(predicted_output))
                        print()

                    train_loss_mean = torch.tensor(train_losses).mean()
                    valid_loss_mean = torch.tensor(valid_losses).mean()
                    print('{}: Sentences Processed: {}/{} / Steps {} : Train:{:.4f}, Valid:{:.4f}({:.4f})'.format(i, cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0], len(cl_direct_noise_train), current_training_step, train_loss_mean, valid_loss_mean, best_valid_loss))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['finetune']['best_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            if in_multigpu_mode:
                                to_save = {
                                    'current_iterating_idx': cl_direct_noise_train.current_iterating_idx -
                                                             t_noised_batch.shape[0],
                                    'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                                    'model_state_dict': model.module.model.state_dict(),
                                    'optim_state_dict': optimizer.state_dict(),
                                    'best_valid_loss': valid_loss_mean,
                                    'current_training_step': current_training_step
                                }
                                if config['optimizer']['scheduler']['use'] == 'one_cycle':
                                    to_save['scheduler_state_dict'] = scheduler.state_dict()
                            else:
                                to_save = {
                                    'current_iterating_idx': cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                    'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                                    'model_state_dict': model.state_dict(),
                                    'optim_state_dict': optimizer.state_dict(),
                                    'best_valid_loss': valid_loss_mean,
                                    'current_training_step': current_training_step
                                }
                                if config['optimizer']['scheduler']['use'] == 'one_cycle':
                                    to_save['scheduler_state_dict'] = scheduler.state_dict()
                            torch.save(to_save, out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                    save_fn = os.path.expandvars(config['finetune']['current_model_save_fn'])
                    with open(save_fn, 'wb') as out_file:
                        if in_multigpu_mode:
                            to_save = {
                                'current_iterating_idx': cl_direct_noise_train.current_iterating_idx -
                                                         t_noised_batch.shape[0],
                                'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                                'model_state_dict': model.module.model.state_dict(),
                                'optim_state_dict': optimizer.state_dict(),
                                'best_valid_loss': best_valid_loss,
                                'current_training_step': current_training_step
                            }
                            if config['optimizer']['scheduler']['use'] == 'one_cycle':
                                to_save['scheduler_state_dict'] = scheduler.state_dict()
                        else:
                            to_save = {
                                'current_iterating_idx': cl_direct_noise_train.current_iterating_idx -
                                                         t_noised_batch.shape[0],
                                'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                                'model_state_dict': model.state_dict(),
                                'optim_state_dict': optimizer.state_dict(),
                                'best_valid_loss': best_valid_loss,
                                'current_training_step': current_training_step
                            }
                            if config['optimizer']['scheduler']['use'] == 'one_cycle':
                                to_save['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(to_save, out_file)

                train_losses.clear()
                model.train()

            if not max_reached:
                optimizer.zero_grad()
                if in_multigpu_mode:
                    loss = model(t_bos_trunc, t_noised_batch, t_eos_trunc, t_input_key_mask, t_output_key_mask, None)
                    loss = loss.mean()
                else:
                    out = model(t_noised_batch, t_eos_trunc, t_input_key_mask, t_output_key_mask, None)
                    loss = criterion(out.contiguous().view(-1, len(vocab)), t_bos_trunc.view(-1))
                loss.backward()
                if config['optimizer']['grad_clip_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip_norm'])
                train_losses.append(loss.item())
                optimizer.step()
                current_training_step += 1
                if config['optimizer']['scheduler']['use'] == 'one_cycle':
                    scheduler.step()
            else:
                break


elif config['mode'] == 'inference':
    if config['inference']['force_cpu']:
        device = 'cpu'

    if config['inference']['preprocess']['activate']:
        import spacy
        import fastBPE

        nlp = spacy.load('en_core_web_sm',
                         disable=['tagger', 'parser', 'ner', 'entity_linker', 'textcat', 'entity_ruler', 'sentencizer',
                                  'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
        codes_fn = os.path.expandvars(config['inference']['preprocess']['bpe_codes_fn'])
        bpe_vocab_fn = os.path.expandvars(config['inference']['preprocess']['bpe_vocab_fn'])
        bpe = fastBPE.fastBPE(codes_fn, bpe_vocab_fn)

    vocab_path = os.path.expandvars(config['inference']['h5']['vocab'])
    with h5py.File(vocab_path, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
        else:
            vocab_special_chars = []

    ft_corpus_path = os.path.expandvars(config['inference']['h5']['ft_corpus'])
    cl = StreamingH5CorpusLoader.load_and_split(
        ft_corpus_path,
        use_split_id=config['inference']['h5']['ft_corpus_split'],
        forced_vocab=(vocab, vocab_special_chars)
    )[0]

    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )

    pretrained_mdl_path = os.path.expandvars(config['inference']['pretrained_model'])
    with open(pretrained_mdl_path, 'rb') as in_file:
        loaded_data = torch.load(in_file, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'])
    model.to(device)
    model.eval()

    source_input_fn = os.path.expandvars(config['inference']['source_fn'])
    hyp_output_fn = os.path.expandvars(config['inference']['hyp_fn'])
    if config['inference']['output_buffering']:
        buffering = -1
    else:
        buffering = 1
    with open(source_input_fn, 'r') as in_f:
        with open(hyp_output_fn, 'w', buffering=buffering) as out_f:
            for li, line in enumerate(in_f):
                if li < config['inference']['line_offset']:
                    continue
                line = line.strip()
                if config['inference']['preprocess']['activate']:
                    line = ' '.join([t.text for t in nlp(line)])
                    line = bpe.apply([line])[0]
                    line = line.lower()
                print('IN  : {}'.format(line))
                if config['inference']['max_len'] > 0 and len(line.split(' ')) > config['inference']['max_len']:
                    print('TOO LONG')
                    continue
                encoded = cl.encode_sentence(line).to(device)

                beam_decoded = model.beam_decode_2(
                    encoded,
                    torch.tensor([cl.bos_idx], dtype=torch.long).to(device),
                    beam_width=config['inference']['beam_width'],
                    max_len=int(encoded.shape[0] * config['inference']['max_len_scale']),
                    end_token=cl.eos_idx,
                    noising_beta=config['inference']['noising_beta'],
                    temperature=config['inference']['temperature'],
                    top_only=False,
                    device=device
                )

                decoded_sentence = cl.decode_tensor(beam_decoded[0])[0]
                if config['inference']['remove_bpe_placeholder']:
                    decoded_sentence = decoded_sentence.replace("@@ ", "")
                if config['inference']['output_parallel']:
                    out_f.write('{} <split> {}\n'.format(line, decoded_sentence))
                else:
                    out_f.write('{}\n'.format(decoded_sentence))
                print('OUT : {}'.format(decoded_sentence))


elif config['mode'] == 'debug':

    vocab_path = 'temp/datasets/bcu_enwiki_30000_bpe_vocab.h5'
    with h5py.File(vocab_path, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
        else:
            vocab_special_chars = []

    ft_corpus_path = 'temp/datasets/gec_combined_real_clean.bpe.h5'
    cl = StreamingH5CorpusLoader.load_and_split(
        ft_corpus_path,
        use_split_id=0,
        forced_vocab=(vocab, vocab_special_chars),
        device=device
    )[0]
    shuffler = SentenceShuffler.chunk_shuffler()
    par_ds = StreamingParallelDataset(cl)
    noise_ds = StreamingCANoiseDataset(cl, shuffler, 0., 0., 0., 1., 0., 0.)
    ds = StreamingChainedDataset(cl, [par_ds, noise_ds])
    b = next(iter(ds))


    import spacy
    import fastBPE


    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner', 'entity_linker', 'textcat', 'entity_ruler', 'sentencizer', 'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
    bpe = fastBPE.fastBPE('temp/datasets/bcu_enwiki.30000.codes', 'temp/datasets/bcu_enwiki_spacy.30000.bpe.vocab')
    # device = 'cpu'



    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers'],
        config['TransformerS2S']['activation']
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    optimizer = optim.Adam(model.parameters(),
                           lr=config['optimizer']['adam']['lr'],
                           betas=(config['optimizer']['adam']['beta_1'], config['optimizer']['adam']['beta_2']),
                           eps=config['optimizer']['adam']['eps'])

    if config['optimizer']['scheduler']['use'] == 'one_cycle':
        pct = config['optimizer']['scheduler']['one_cycle']['warmup_steps'] / \
              config['optimizer']['scheduler']['one_cycle']['total_steps']
        print('Scheduler Pct: {:%}'.format(pct))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config['optimizer']['scheduler']['one_cycle']['max_lr'],
            div_factor=config['optimizer']['scheduler']['one_cycle']['initial_lr_div'],
            final_div_factor=config['optimizer']['scheduler']['one_cycle']['final_lr_div'],
            total_steps=config['optimizer']['scheduler']['one_cycle']['total_steps'],
            anneal_strategy=config['optimizer']['scheduler']['one_cycle']['anneal_strategy'],
            pct_start=pct,
            last_epoch=-1,
            cycle_momentum=False
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num Params : {:,}".format(pytorch_total_params))


    pretrained_mdl_path = 'temp/models/best_bcuenwiki_gec_gbl_ft.pkl'
    with open(pretrained_mdl_path, 'rb') as in_file:
        loaded_data = torch.load(in_file, map_location=device)
        model.load_state_dict(loaded_data['model_state_dict'])
    model.to(device)
    model.eval()

    ft_corpus_path = 'temp/datasets/gec_combined_real_clean.bpe.h5'
    cl = StreamingH5CorpusLoader.load_and_split(
        ft_corpus_path,
        use_split_id=0,
        forced_vocab=(vocab, vocab_special_chars),
        device=device
    )[0]

    test = [((b,e), cl.corpus[b:e]) for b, e in cl.sentences[(cl.sentences[:,1] - cl.sentences[:,0] == 5).nonzero().squeeze(1)] if torch.tensor([35,17,1801,9,6],dtype=torch.int).allclose(cl.corpus[b:e])]

    with open('/run/media/samuel/Data/UdeM/Recherche/Corpus/SimpleWiki/simple_wiki.noisy.train', 'w') as out_file:
        with open('/run/media/samuel/Data/UdeM/Recherche/Corpus/SimpleWiki/simple_wiki.sent.clean', 'r') as in_file:
            for i, line in enumerate(in_file):
                line = "the team with the most points at the end of the game wins ."
                line = line.strip().lower()
                line = ' '.join([t.text for t in nlp(line)])
                line = bpe.apply([line])[0]
                print('{}: {}'.format(i, line))
                if len(line.split())> 125:
                    print('\tTOO LONG')
                    continue
                encoded = cl.encode_sentence(line).to(device)
                beam_decoded = model.beam_decode_2(
                    encoded,
                    torch.tensor([cl.bos_idx], dtype=torch.long).to(device),
                    beam_width=8,
                    max_len=int(encoded.shape[0] * 1.5),
                    end_token=cl.eos_idx,
                    noising_beta=0.3,
                    topmost_noising=False,
                    temperature=100.,
                    top_only=False,
                    device=device
                )
                for bd in beam_decoded:
                    sent = cl.decode_tensor(bd)
                    if line == sent[0]:
                        continue
                    else:
                        print('\t{}'.format(sent[0]))
                        out_file.write('{} <split> {}\n'.format(line, sent[0]))
                        break


            # encoded_test = cl.encode_sentence(test)
            # beam_decoded = model.beam_decode(
            #                 encoded_test,
            #                 torch.tensor([cl.bos_idx], dtype=torch.long),
            #                 beam_width=2,
            #                 max_len=encoded_test.shape[0]*2,
            #                 end_token=cl.eos_idx,
            #                 noising_beta=0.1,
            #                 top_only=False,
            #                 device=device
            #             )
            # for bd in beam_decoded:
            #     print(cl.decode_tensor(bd))
