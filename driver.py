import argparse
import yaml
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from stylecorrection.loaders.corpus import *
from stylecorrection.models.transformer import TransformerS2S
from stylecorrection.utils.evaluation import compute_scores

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
params = parser.parse_args()

with open(params.config, 'r') as in_file:
    config = yaml.load(in_file, Loader=yaml.FullLoader)

# with h5py.File('temp/models/bcu_enwiki_50k_mf2_s0_vocab.h5') as h5_file:
#     vocab = h5_file['vocab'][:]
#
# cl = H5CorpusLoader.load_and_split(
#     'temp/datasets/obw.h5',
#     use_split_id=0,
#     forced_vocab=vocab
# )
#
# model = TransformerS2S(
#     len(vocab),
#     512,
#     8,
#     4096,
#     6,
#     6
# )
#
# with open('temp/models/bcu_enwiki_dn_obw_CA_3_ft.pkl', 'rb') as in_file:
#     loaded_data = torch.load(in_file, map_location=device)
#     model.load_state_dict(loaded_data)
#
# model.eval()
#
# for ds, cs in zip(config['sample_corrections']['dirty'] + ["it is the first time for me to come here ."], config['sample_corrections']['clean']+ ["i hope to hear from you soon ."]):
#     test_sentence = cl.encode_sentence(ds)
#
#     with torch.no_grad():
#         res = model.beam_decode(
#             test_sentence,
#             torch.tensor([cl.bos_idx], dtype=torch.long),
#             beam_width=5,
#             max_len=len(test_sentence)+5,
#             end_token=cl.eos_idx
#         )
#
#     decoded = cl.decode_tensor(torch.tensor(res, dtype=torch.long))
#     # scores = compute_scores([decoded[0].split(' ')[1:-1]], [cs.split(' ')])
#     print('[{}] -> [{}] ({})'.format(ds, decoded, cs))
#     # print(scores)
#
# exit()

if config['mode'] == 'hd5_gen':
    print('Creating hd5 dataset...')
    h5_fn = os.path.expandvars(config['hd5_gen']['h5_fn'])
    H5CorpusLoader.create_from_compressed(
        h5_fn,
        os.path.expandvars(config['hd5_gen']['corpus_tar_gz']),
        lambda x: x.strip().split(' '),
        lambda x: x.lower(),
        config['hd5_gen']['topk'],
        config['hd5_gen']['max_len']
    )
    print('DONE')

elif config['mode'] == 'gen_split':
    print('Generating train/valid split...')
    h5_fn = os.path.expandvars(config['gen_split']['h5_fn'])
    H5CorpusLoader.generate_split(
        h5_fn,
        config['gen_split']['valid_ratio']
    )
    print('DONE')

elif config['mode'] == 'vocab_dump':
    print('Starting vocab dumping...')
    corpus_h5_fn = os.path.expandvars(config['vocab_dump']['corpus_h5_fn'])
    vocab_h5_fn = os.path.expandvars(config['vocab_dump']['vocab_h5_fn'])
    cl = H5CorpusLoader.load_and_split(
        corpus_h5_fn,
        config['vocab_dump']['valid_split_id'],
        config['vocab_dump']['vocab_topk'],
        config['vocab_dump']['min_freq'],
        device=device
    )
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
    cl_train, cl_valid = StreamingH5CorpusLoader.load_and_split(
        corpus_h5_fn,
        use_split_id=config['pretrain']['hd5']['valid_split_id'],
        forced_vocab=vocab,
        device=device
    )
    if config['pretrain']['algo'] == 'bart':
        print('Using text infilling (BART)')
        pds_train = StreamingBARTPretrainingDataset(cl_train, tokens_per_batch=config['pretrain']['tpb'], device=device)
        pds_valid = StreamingBARTPretrainingDataset(cl_valid, tokens_per_batch=config['pretrain']['tpb'], device=device)
    else:
        pds_train = StreamingMASSPretrainingDataset(cl_train, tokens_per_batch=config['pretrain']['tpb'], device=device)
        pds_valid = StreamingMASSPretrainingDataset(cl_valid, tokens_per_batch=config['pretrain']['tpb'], device=device)

    model = TransformerS2S(
        len(cl_train.vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )


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

    if config['pretrain']['resume_from'] == 'best':
        model_save_fn = os.path.expandvars(config['pretrain']['best_model_save_fn'])
    else:
        model_save_fn = os.path.expandvars(config['pretrain']['current_model_save_fn'])
    if os.path.exists(model_save_fn):
        with open(model_save_fn, 'rb') as data_file:
            print('Loading from {}'.format(model_save_fn))
            loaded_data = torch.load(data_file, map_location='cpu')
            cl_train.current_iterating_idx = loaded_data['current_iterating_idx']
            cl_train.current_iterating_order = loaded_data['current_iterating_order']
            cl_train.generate_iterating_order = False
            model.load_state_dict(loaded_data['model_state_dict'])
            optimizer.load_state_dict(loaded_data['optimizer_state_dict'])

    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=cl_train.pad_idx).to(device)

    train_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0

    for i in range(config['pretrain']['max_epoch']):
        model.train()
        for tbi, (t_enc_in, t_enc_in_key_mask, t_dec_out, t_dec_in, t_dec_in_key_mask, t_offsets) in enumerate(pds_train):

            if tbi % config['eval']['interval'] == 0:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets) in enumerate(pds_valid):
                        out = model(v_enc_in, v_dec_in, v_enc_in_key_mask, v_dec_in_key_mask, v_offsets)
                        loss = criterion(out.contiguous().view(-1, len(cl_valid.vocab)), v_dec_out.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_enc_in, v_enc_in_key_mask, v_dec_out, v_dec_in, v_dec_in_key_mask, v_offsets = next(iter(pds_valid))
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
                    print('{}: Sentences Processed {}/{} : Train:{:.4f}, Valid:{:.4f}'.format(i, cl_train.current_iterating_idx - t_enc_in.shape[0], len(cl_train), train_loss_mean, valid_loss_mean))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['pretrain']['best_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            to_save = {
                                'current_iterating_idx': cl_train.current_iterating_idx - t_enc_in.shape[0],
                                'current_iterating_order': cl_train.current_iterating_order,
                                'model_state_dict': model.state_dict(),
                                'optim_state_dict': optimizer.state_dict()
                            }
                            torch.save(to_save, out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                    save_fn = os.path.expandvars(config['pretrain']['current_model_save_fn'])
                    with open(save_fn, 'wb') as out_file:
                        to_save = {
                            'current_iterating_idx': cl_train.current_iterating_idx - t_enc_in.shape[0],
                            'current_iterating_order': cl_train.current_iterating_order,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict()
                        }
                        torch.save(to_save, out_file)

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
            loss = criterion(out.contiguous().view(-1, len(cl_train.vocab)), t_dec_out.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

        if patience_counter > config['pretrain']['valid_patience']:
            print('Patience threshold reached')
            break
    print('DONE')

elif config['mode'] == 'finetune_streaming':
    h5_fn_finetune = os.path.expandvars(config['finetune']['hd5']['finetune']['h5_fn'])
    vocab_h5_fn = os.path.expandvars(os.path.expandvars(config['finetune']['hd5']['vocab']['h5_fn']))
    with h5py.File(vocab_h5_fn, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
    cl_direct_noise_train, cl_direct_noise_valid = StreamingH5CorpusLoader.load_and_split(
        h5_fn_finetune,
        use_split_id=config['finetune']['hd5']['finetune']['valid_split_id'],
        forced_vocab=vocab,
        smoothing_alpha=config['finetune']['hd5']['finetune']['smoothing_alpha']
    )
    if config['finetune']['dataset']['to_use'] == 'ca':
        dnds_train = StreamingCANoiseDataset(cl_direct_noise_train,
                                             replace_prob=config['finetune']['dataset']['ca']['replace_prob'],
                                             del_prob=config['finetune']['dataset']['ca']['del_prob'],
                                             ins_prob=config['finetune']['dataset']['ca']['ins_prob'],
                                             keep_prob=config['finetune']['dataset']['ca']['keep_prob'],
                                             mask_prob=config['finetune']['dataset']['ca']['mask_prob'],
                                             sigma=config['finetune']['dataset']['ca']['sigma'],
                                             tokens_per_batch=config['finetune']['dataset']['ca']['tpb'],
                                             device=device)
        dnds_valid = StreamingCANoiseDataset(cl_direct_noise_valid,
                                             replace_prob=config['finetune']['dataset']['ca']['replace_prob'],
                                             del_prob=config['finetune']['dataset']['ca']['del_prob'],
                                             ins_prob=config['finetune']['dataset']['ca']['ins_prob'],
                                             keep_prob=config['finetune']['dataset']['ca']['keep_prob'],
                                             mask_prob=config['finetune']['dataset']['ca']['mask_prob'],
                                             sigma=config['finetune']['dataset']['ca']['sigma'],
                                             tokens_per_batch=config['finetune']['dataset']['ca']['tpb'],
                                             device=device)
    else:
        pass
        # dnds = DirectNoiseDataset(cl_direct_noise,
        #                           del_prob=config['finetune']['dataset']['dn']['del_prob'],
        #                           ins_prob=config['finetune']['dataset']['dn']['ins_prob'],
        #                           keep_prob=config['finetune']['dataset']['dn']['keep_prob'],
        #                           mask_prob=config['finetune']['dataset']['dn']['mask_prob'],
        #                           device=device)

    model = TransformerS2S(
        len(vocab),
        config['TransformerS2S']['emb_dim'],
        config['TransformerS2S']['n_head'],
        config['TransformerS2S']['ff_dim'],
        config['TransformerS2S']['num_enc_layers'],
        config['TransformerS2S']['num_dec_layers']
    )

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

    if config['finetune']['resume_from'] == 'best':
        model_save_fn = os.path.expandvars(config['finetune']['best_model_save_fn'])
    else:
        model_save_fn = os.path.expandvars(config['finetune']['current_model_save_fn'])
    if os.path.exists(model_save_fn):
        with open(model_save_fn, 'rb') as data_file:
            print('Loading from {}'.format(model_save_fn))
            loaded_data = torch.load(data_file, map_location='cpu')
            cl_direct_noise_train.current_iterating_idx = loaded_data['current_iterating_idx']
            cl_direct_noise_train.current_iterating_order = loaded_data['current_iterating_order']
            cl_direct_noise_train.generate_iterating_order = False
            model.load_state_dict(loaded_data['model_state_dict'])
            optimizer.load_state_dict(loaded_data['optimizer_state_dict'])


    if config['multi_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    model_fn = os.path.expandvars(config['finetune']['pretrain_model_fn'])
    with open(model_fn, 'rb') as in_file:
        model.load_state_dict(torch.load(in_file, map_location=device))

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=cl_direct_noise_train.pad_idx).to(device)
    train_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    for i in range(config['finetune']['max_epoch']):
        model.train()
        for tbi, (t_noised_batch, t_input_key_mask, t_eos_trunc, t_bos_trunc, t_output_key_mask) in enumerate(dnds_train):

            if tbi % config['eval']['interval'] == 0:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (v_noised_batch, v_input_key_mask, v_eos_trunc, v_bos_trunc, v_output_key_mask) in enumerate(dnds_valid):
                        out = model(v_noised_batch, v_eos_trunc, v_input_key_mask, v_output_key_mask, None)
                        loss = criterion(out.contiguous().view(-1, len(cl_direct_noise_valid.vocab)), v_bos_trunc.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    v_noised_batch, v_input_key_mask, v_eos_trunc, v_bos_trunc, v_output_key_mask = next(iter(dnds_valid))
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
                    print('{}: Sentences Processed: {}/{}  Train:{:.4f}, Valid:{:.4f}'.format(i, cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0], len(cl_direct_noise_train), train_loss_mean, valid_loss_mean))

                    if valid_loss_mean < best_valid_loss:
                        save_fn = os.path.expandvars(config['finetune']['best_model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            to_save = {
                                'current_iterating_idx': cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                                'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                                'model_state_dict': model.state_dict(),
                                'optim_state_dict': optimizer.state_dict()
                            }
                            torch.save(to_save, out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                    save_fn = os.path.expandvars(config['finetune']['current_model_save_fn'])
                    with open(save_fn, 'wb') as out_file:
                        to_save = {
                            'current_iterating_idx': cl_direct_noise_train.current_iterating_idx - t_noised_batch.shape[0],
                            'current_iterating_order': cl_direct_noise_train.current_iterating_order,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict()
                        }
                        torch.save(to_save, out_file)

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(t_noised_batch, t_eos_trunc, t_input_key_mask, t_output_key_mask, None)
            loss = criterion(out.contiguous().view(-1, len(cl_direct_noise_train.vocab)), t_bos_trunc.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
