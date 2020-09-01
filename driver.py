import argparse
import yaml
import os
import torch.optim as optim
from typing import Optional
from stylecorrection.utils.GLEU import GLEU
from stylecorrection.loaders.corpus import *
from stylecorrection.models.wrappers import *
from stylecorrection.models.transformer import TransformerS2S
from stylecorrection.operation.finetune import FinetuneStreamingOperation
from stylecorrection.operation.pretrain import PretrainOperation
from stylecorrection.operation.inference import InferenceOperation
from stylecorrection.operation.hd5gen import Hd5GenOperation, VocabDumpOperation
from stylecorrection.operation.eval import ManualEvaluationOperation
from stylecorrection.utils.config import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
params = parser.parse_args()

oc_conf = OneCycleConfig(0., 0., 0., 0, 0)
s_conf = SchedulerConfig('none', oc_conf)
yaml.add_representer(OneCycleConfig, OneCycleConfig.to_yaml)
yaml.add_representer(SchedulerConfig, SchedulerConfig.to_yaml)
yaml.dump(s_conf)

with open(params.config, 'r') as in_file:
    yaml.add_constructor('!TransformerS2S', TransformerConfig.from_yaml)
    yaml.add_constructor('!eval', EvaluationConfig.from_yaml)
    yaml.add_constructor('!global', GlobalConfig.from_yaml)
    yaml.add_constructor('!hd5_gen', GenerateHd5Config.from_yaml)
    yaml.add_constructor('!preprocess', PreprocessConfig.from_yaml)
    yaml.add_constructor('!training_max', TrainingMaxConfig.from_yaml)
    yaml.add_constructor('!optimizer', OptimizerConfig.from_yaml)
    yaml.add_constructor('!scheduler', SchedulerConfig.from_yaml)
    yaml.add_constructor('!one_cycle', OneCycleConfig.from_yaml)
    yaml.add_constructor('!adam', AdamConfig.from_yaml)
    yaml.add_constructor('!sgd', SgdConfig.from_yaml)
    yaml.add_constructor('!inference', InferenceConfig.from_yaml)
    yaml.add_constructor('!hd5_dataset', Hd5DatasetConfig.from_yaml)
    yaml.add_constructor('!manual_eval', ManualEvalConfig.from_yaml)
    yaml.add_constructor('!sample_corrections', SampleCorrectionsConfig.from_yaml)
    yaml.add_constructor('!finetune', FinetuneConfig.from_yaml)
    yaml.add_constructor('!model_files', ModelFilenameConfig.from_yaml)
    yaml.add_constructor('!finetune_dataset', FinetuneDatasetConfig.from_yaml)
    yaml.add_constructor('!parallel', ParallelDatasetConfig.from_yaml)
    yaml.add_constructor('!ca', CaDatasetConfig.from_yaml)
    yaml.add_constructor('!pretrain', PretrainConfig.from_yaml)
    yaml.add_constructor('!vocab_dump', VocabDumpConfig.from_yaml)
    yaml.add_constructor('!gleu', GleuConfig.from_yaml)
    yaml.add_constructor('!all_config', AllConfig.from_yaml)
    config: AllConfig = yaml.load(in_file, Loader=yaml.Loader)

if config.global_conf.mode == OperationModes.EVAL:
    operation = ManualEvaluationOperation(
        config.global_conf,
        config.transformer_s2s,
        config.manual_eval,
        device
    )

    operation.run()

if config.global_conf.mode == OperationModes.HD5_GEN or config.global_conf.mode == OperationModes.GEN_SPLIT:
    operation = Hd5GenOperation(
        config.global_conf,
        config.hd5_gen,
        device
    )
    operation.run()

elif config.global_conf.mode == OperationModes.VOCAB_DUMP:
    operation = VocabDumpOperation(
        config.global_conf,
        config.vocab_dump,
        device
    )
    operation.run()

elif config.global_conf.mode == OperationModes.PRETRAIN:
    operation = PretrainOperation(
        config.global_conf,
        config.eval,
        config.transformer_s2s,
        config.pretrain,
        config.optimizer,
        device
    )
    operation.run()

elif config.global_conf.mode == OperationModes.FINETUNE:
    operation = FinetuneStreamingOperation(config, device)
    operation.run()

elif config.global_conf.mode == OperationModes.INFERENCE:
    operation = InferenceOperation(config, device)
    operation.run()

elif config.global_conf.mode == OperationModes.DEBUG:

    vocab_path = 'temp/datasets/bcu_enwiki_30000_bpe_vocab.h5'
    with h5py.File(vocab_path, 'r') as h5_file:
        vocab = h5_file['vocab'][:]
        if 'additional_special_tokens' in h5_file['vocab'].attrs:
            additional_special_tokens = h5_file['vocab'].attrs['additional_special_tokens']
            vocab_special_chars = vocab[5:5 + additional_special_tokens].tolist()
        else:
            vocab_special_chars = []

    ft_corpus_path = 'temp/datasets/gec_combined_real_gen800k_mp_mn_homo.bpe.h5'
    cl = StreamingH5CorpusLoader.load_and_split(
        ft_corpus_path,
        group_by_len=True,
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

    nlp = spacy.load('en',
                     disable=['tagger', 'parser', 'ner', 'entity_linker', 'textcat', 'entity_ruler', 'sentencizer',
                              'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
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

    test = [((b, e), cl.corpus[b:e]) for b, e in
            cl.sentences[(cl.sentences[:, 1] - cl.sentences[:, 0] == 5).nonzero().squeeze(1)] if
            torch.tensor([35, 17, 1801, 9, 6], dtype=torch.int).allclose(cl.corpus[b:e])]

    with open('/run/media/samuel/Data/UdeM/Recherche/Corpus/SimpleWiki/simple_wiki.noisy.train', 'w') as out_file:
        with open('/run/media/samuel/Data/UdeM/Recherche/Corpus/SimpleWiki/simple_wiki.sent.clean', 'r') as in_file:
            for i, line in enumerate(in_file):
                line = "the team with the most points at the end of the game wins ."
                line = line.strip().lower()
                line = ' '.join([t.text for t in nlp(line)])
                line = bpe.apply([line])[0]
                print('{}: {}'.format(i, line))
                if len(line.split()) > 125:
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

