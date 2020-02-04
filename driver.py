import torch
import torch.nn as nn
import torch.optim as optim
from stylecorrection.loaders.corpus import CorpusLoader, PretrainingDataset, DirectNoiseDataset
from stylecorrection.models.transformer import TransformerS2S

bl = CorpusLoader(lambda x: x.strip().split(' '), topk=4000, vocab_topk=30000)
bl.extract_from_archive('temp/datasets/BookCorpus_unique.tar.gz')
pds = PretrainingDataset(bl)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = TransformerS2S(len(bl.vocab), 120, device=device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=bl.pad_idx).to(device)

for i in range(100):
    train_losses = []
    for enc_in, enc_in_key_mask, dec_out, dec_in, dec_in_key_mask, offsets in pds():
        optimizer.zero_grad()
        out = model(enc_in, dec_in, enc_in_key_mask, dec_in_key_mask, offsets)
        loss = criterion(out.contiguous().view(-1, len(bl.vocab)), dec_out.view(-1))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    print('Iteration {} : {:.4f}'.format(i, torch.tensor(train_losses).mean()))
