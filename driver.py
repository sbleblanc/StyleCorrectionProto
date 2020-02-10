import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from stylecorrection.loaders.corpus import  PretrainingDataset, H5CorpusLoader
from stylecorrection.models.transformer import TransformerS2S

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
params = parser.parse_args()

with open(params.config, 'r') as in_file:
    config = yaml.load(in_file, Loader=yaml.FullLoader)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

h5_fn = os.path.expandvars(config['h5_fn'])

if config['mode'] == 'hd5_gen':
    print('Creating hd5 dataset...')
    H5CorpusLoader.create_from_compressed(
        h5_fn,
        os.path.expandvars(config['H5CorpusLoader_create']['corpus_tar_gz']),
        lambda x: x.strip().split(' '),
        None,
        config['H5CorpusLoader_create']['topk'],
        config['H5CorpusLoader_create']['max_len']
    )
    print('DONE')

elif config['mode'] == 'gen_split':
    print('Generating train/valid split...')
    H5CorpusLoader.generate_split(
        h5_fn,
        config['H5CorpusLoader_gen_split']['valid_ratio']
    )
    print('DONE')

elif config['mode'] == 'pretrain':
    print('Starting Pretraining...')

    cl = H5CorpusLoader.load_and_split(
        h5_fn,
        config['H5CorpusLoader_load']['valid_split_id'],
        config['H5CorpusLoader_load']['vocab_topk'],
        config['H5CorpusLoader_load']['min_freq'],
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
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['adam']['lr'],
                               betas=(config['adam']['beta_1'], config['adam']['beta_2']),
                               eps=config['adam']['eps'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['sgd']['lr'],
                              momentum=config['sgd']['momentum'],
                              weight_decay=config['sgd']['weight_decay'],
                              nesterov=config['sgd']['nesterov'])
    criterion = nn.CrossEntropyLoss(ignore_index=cl.pad_idx).to(device)

    for i in range(config['pretraining']['max_epoch']):
        train_losses = []

        best_valid_loss = float('inf')
        patience_counter = 0
        bs = config['pretraining']['bs']

        model.train()
        for tbi, (enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets) in enumerate(pds(bs=bs)):

            if tbi % config['eval']['interval'] == 0:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for vbi, (enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets) in enumerate(pds(bs=bs, which='valid')):
                        out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
                        loss = criterion(out.contiguous().view(-1, len(cl.vocab)), dec_out.view(-1))
                        valid_losses.append(loss.item())
                        if vbi == config['eval']['num_valid_batch']:
                            break
                    enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets = next(pds(bs=1, which='valid'))
                    out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
                    enc_input = cl.decode_tensor(enc_in)
                    expected_output = cl.decode_tensor(dec_out)
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
                        with open(config['pretraining']['model_save_fn'], 'wb') as out_file:
                            torch.save(model.state_dict(), out_file)
                        patience_counter = 0
                        best_valid_loss = valid_loss_mean
                    else:
                        patience_counter += 1

                train_losses.clear()
                model.train()

            optimizer.zero_grad()
            out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
            loss = criterion(out.contiguous().view(-1, len(cl.vocab)), dec_out.view(-1))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()

        if patience_counter > config['pretraining']['valid_patience']:
            print('Patience threshold reached')
            break
    print('DONE')