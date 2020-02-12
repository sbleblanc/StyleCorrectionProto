import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from stylecorrection.loaders.corpus import  PretrainingDataset, H5CorpusLoader, DirectNoiseDataset
from stylecorrection.models.transformer import TransformerS2S

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# test_sentence = cl.encode_sentence('kyle martino <mask> born <mask> <mask> the <mask> <mask> in atlanta , georgia .')
#
# res = model.beam_decode(
#     test_sentence,
#     torch.tensor([cl.mask_idx], dtype=torch.long),
#     beam_width=5,
#     max_len=7,
#     position_offset=3
# )
#
# cl.decode_tensor(torch.tensor(res, dtype=torch.long))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
params = parser.parse_args()

with open(params.config, 'r') as in_file:
    config = yaml.load(in_file, Loader=yaml.FullLoader)

if config['mode'] == 'hd5_gen':
    print('Creating hd5 dataset...')
    h5_fn = os.path.expandvars(config['hd5_gen']['h5_fn'])
    H5CorpusLoader.create_from_compressed(
        h5_fn,
        os.path.expandvars(config['hd5_gen']['corpus_tar_gz']),
        lambda x: x.strip().split(' '),
        None,
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

elif config['mode'] == 'pretrain':
    print('Starting Pretraining...')
    h5_fn = os.path.expandvars(config['pretrain']['hd5']['h5_fn'])
    cl = H5CorpusLoader.load_and_split(
        h5_fn,
        config['pretrain']['hd5']['valid_split_id'],
        config['pretrain']['hd5']['vocab_topk'],
        config['pretrain']['hd5']['min_freq'],
        device=device
    )
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
    bs = config['pretraining']['bs']

    for i in range(config['pretraining']['max_epoch']):
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
                        save_fn = os.path.expandvars(config['pretraining']['model_save_fn'])
                        with open(save_fn, 'wb') as out_file:
                            torch.save(model.state_dict(), out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(t_enc_in, t_dec_in, t_enc_in_key_mask, t_dec_in_key_mask, t_offsets)
            loss = criterion(out.contiguous().view(-1, len(cl.vocab)), t_dec_out.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

        if patience_counter > config['pretraining']['valid_patience']:
            print('Patience threshold reached')
            break
    print('DONE')

elif config['mode'] == 'finetune':
    h5_fn_pretrain = os.path.expandvars(config['finetune']['hd5']['pretrained']['h5_fn'])
    h5_fn_finetune = os.path.expandvars(config['finetune']['hd5']['finetune']['h5_fn'])
    cl_pretrain = H5CorpusLoader.load_and_split(
        h5_fn_pretrain,
        config['finetune']['hd5']['pretrained']['valid_split_id'],
        config['finetune']['hd5']['pretrained']['vocab_topk'],
        config['finetune']['hd5']['pretrained']['min_freq'],
        device=device
    )
    cl_direct_noise = H5CorpusLoader.load_and_split(
        h5_fn_finetune,
        use_split_id=config['finetune']['hd5']['finetune']['valid_split_id'],
        forced_vocab=cl_pretrain.vocab,
        smoothing_alpha=config['finetune']['hd5']['finetune']['smoothing_alpha']
    )
    dnds = DirectNoiseDataset(cl_direct_noise, device=device)
    model = TransformerS2S(
        len(cl_pretrain.vocab),
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
        model.load_state_dict(torch.load(in_file))

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
    bs = config['pretraining']['bs']
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
