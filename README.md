# Sentence Style Correction

This is code for the experimentations on style correction for english text. This code implements : 
-  Corpus conversion to hd5 files for better memory efficiency
- Transformer sequence to sequence model
- [MASS](https://arxiv.org/abs/1905.02450) and [BART](https://arxiv.org/abs/1910.13461) pretraining
- [Beam decoding noise](https://www.aclweb.org/anthology/N18-1057.pdf) during inference

This projet is not intended to be used as a general solution for this problem but it should be easily adaptable to be more flexible. The pretraining corpus is intended to be tokenized using SpaCy and BPE should be applied using fastBPE. 

With this project, you will be able to pretrain a Transformer sequence to sequence model using either the MASS or BART technique. This pretrained model will then be able to be finetuned using either a parallel dataset or a clean dataset with dynamic error generation using configurable random noise. Inference can then be done on the final model. If an error generating model has been trained (i.e. dirty to clean), noising can be applied to the beam decoding scores during inference to have a more diverse set of errors.

## Requirements
- Python 3.8
- torch
- SpaCy
- fastBPE
- h5py
- yaml

## Intended flow of execution
The whole process is divided in several independent steps that are executed seperatly one at the time. A normal ordering of these steps would be :

1. Convert a raw pretraining corpus into a hd5 format
2. Generate a train/validation split on the generated hd5 dataset
3. Dump the vocabulary of the generated hd5 dataset based on the train/valid split
4. Repeat steps 1 and 2 for any finetuning datasets
5. Pretrain the model using the preprocessed pretraining hd5 dataset
6. Finetune the model using the preprocessed finetuning hd5 dataset
7. Perform inference using the finetuned model

## Execution Modes

These are the different execution mode that can be used. The specifics of each can be understood in the comments of the [sample configuration file](./config.example.yaml).

### hd5_gen
<ins>Configurations section required</ins> :
- hd5_gen

<ins>Files Required</ins> :
- Raw corpus compressed in a tar.gz

Mode used to convert a raw corpus into the required hd5 format. The source raw corpus should be in a tar.gz archive with the text files at the root. The method was intended for datasets like BookCorpus which are divided in thousands of different files. So, raw datasets **SHOULD NOT** be combined before compressing to tar.gz since each document in the archive is loaded in memory completely. If a document is too large, you will most likely run out of memory. A big document can be split into multiple smaller one before compressing if this is the case. The current implementation assumes the raw corpus to be :
- Sentencized : Sentences must be seperated by a newline character (\n).
- Tokenized : The text should be tokenized with SpaCy
- Byte Pair Encoding applied : BPE should be applied using fastBPE

 It will split words based on spaces. The code can be adapted so that the preprocessing and/or tokenization is done at this step, but keep in mind that the way it is implemented now, the corpus is processed at a sentence level (a sentence being a single split of the whole corpus on newline characters). For example, BPE could be applied at this step, but the codes must be pre-computed (e.g. using fastBPE).

### gen_split
<ins>Configurations section required</ins> :
- hd5_gen

<ins>Files Required</ins> :
- Generated hd5 dataset

Mode used to generate a train/valid split for a dataset in the hd5 format. Up to 20 splits can be generated on a single hd5 dataset. **By default, a new hd5 dataset does not contain a split and won't be usable until at least one is generated.**

### vocab_dump
<ins>Configurations section required</ins> :
- vocab_dump

<ins>Files Required</ins> :
- Generated hd5 dataset

Mode used to dump the vocabulary of a hd5 dataset. At least one train/valid split must be available as the vocabulary is only based off the training portion. This vocabulary will be used when using a different dataset than the one used for pretraining. This will generate a vocabulary file in the hd5 format also.

### pretrain
<ins>Configurations section required</ins> :
- transformer_s2s
- eval
- optimizer
- pretrain

<ins>Files Required</ins> :
- Preprocessed pretraining corpus converted to hd5
- Vocabulary hd5 file

Mode used to pretrain a model. A hd5 dataset with at least one split generated and the dumped vocabulary are required to start the pretraining. At the interval set in the **eval** section, the model will be evaluated on a random subset of the validation set (based on the number of batches set in the configurations). Only a subset is used so that training does not take too long, but the code can be adapted to process the whole validation set. Before the checkpoints are saved, a random sentence is processed from the validation set and the result is shown on screen with the loss metrics.

### finetune
<ins>Configurations section required</ins> :
- transformer_s2s
- eval
- optimizer
- finetune
- (optional) gleu
- (optional) preprocess

<ins>Files Required</ins> :
- Pretrained model (best or current checkpoint)
- Preprocessed finetuning corpus converted to hd5
- Vocabulary hd5 file used for pretraining
- (optional) source and references text files for GLEU evaluation

Mode used to finetune a pretrained model. The transformer model configurations should be the same as during the pretraining. The finetuning corpus should be preprocessed the same way as the pretraining corpus. If GLEU is used to dertermine the best model, the **gleu** section in the configurations should be set. If the *preprocess* option is set for gleu, the **preprocess** section in the configuration must be set. The preprecessing involves tokenizing with SpaCy, applying BPE (using the codes computed with fastBPE for the pretraining) and putting everything in lowercase. The GLEU evaluation was intended to be used with the *dev* set of the [JFLEG evaluation corpus](https://github.com/keisks/jfleg). Before the checkpoints are saved, a random sentence is processed from the validation set and the result is shown on screen with the loss metrics.

If the **parallel** dataset is used, the raw finetuning dataset must be formated correctly before converting to the required hd5 format. Parallel sentences must be combined on the same line with a splitting token in the middle (e.g. helo <split> hello). This token must be specified in the hd5_gen configuration and in the *dataset* section of the finetune configurations. It must also be tokenized and preprocessed like the pretraining dataset.

If the **ca** dataset is used, the raw finetuning dataset is expected to be clean sentences. It must be tokenized and preprocessed like the pretraining dataset before converting to the required hd5 format.

### eval
<ins>Configurations section required</ins> :
- transformer_s2s
- manual_eval

<ins>Files Required</ins> :
- Finetuned model (best or current checkpoint)
- Preprocessed finetuning corpus converted to hd5
- Vocabulary hd5 file used for pretraining

Mode used to correct sentences manually selected. This mode is intended to test the model quickly during training or after. It will translate all the sentences in the list provided in the configurations and output the result on screen. The sentences should already be preprocessed and tokenized like the pretraing corpus (e.g. Spacy tokenization with BPE applied). 


### inference
<ins>Configurations section required</ins> :
- transformer_s2s
- inference
- (optional) preprocess

<ins>Files Required</ins> :
- Finetuned model (best or current checkpoint)
- Preprocessed finetuning corpus converted to hd5
- Vocabulary hd5 file used for pretraining
- Text file with source sentences to correct

Mode used to do inference with a finetuned model. It will process each sentence in the source file and output the best decoded sentence on screen and to the specified text file. If the source file is not preprocessed, the *preprocess* option can be set to True and each sentence will be tokenized with SpaCy, BPE will be applied and the text will be put in lowercase. To preprocess the text differently, either change the code or provide a source text file with preprocessed sentences. Following this [paper](https://www.aclweb.org/anthology/N18-1057.pdf), it is possible to noise the beam decoding scores. This is useful for generating errors that are more diverse if a reverse model has been trained (i.e. dirty to clean). The bigger the *noising_beta* is, the bigger the noise will be. The *temperature* configuration was added to help with the noising since sometimes the best decoding candidate has a score that is a lot better than the other beam candidats, so even with the noise, less probable sentences are not often picked. Basically, the beam decoding scores are divided by the temperature before the softmax is applied. So the bigger the temperature is, the smaller the gap gets between the scores making the noise more effective.
