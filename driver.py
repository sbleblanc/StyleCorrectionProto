import torch
import torch.nn as nn
import torch.optim as optim
from stylecorrection.loaders.corpus import CorpusLoader, PretrainingDataset, DirectNoiseDataset, H5CorpusLoader
from stylecorrection.models.transformer import TransformerS2S

device = "cuda:0" if torch.cuda.is_available() else "cpu"

cl = H5CorpusLoader.load_and_split('temp/datasets/BookCorpus_unique.h5', device=device)
pds = PretrainingDataset(cl, device=device)

model = TransformerS2S(len(cl.vocab), 24, device=device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=cl.pad_idx).to(device)

for i in range(100):
    train_losses = []
    valid_losses = []

    model.train()
    for enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets in pds():
        optimizer.zero_grad()
        out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
        loss = criterion(out.contiguous().view(-1, len(cl.vocab)), dec_out.view(-1))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    model.eval()
    for enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets in pds(which='valid'):
        out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
        loss = criterion(out.contiguous().view(-1, len(cl.vocab)), dec_out.view(-1))
        valid_losses.append(loss.item())

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

    print('Iteration {} : Train:{:.4f}, Valid:{:.4f}'.format(i, torch.tensor(train_losses).mean(), torch.tensor(valid_losses).mean()))
