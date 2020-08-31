# Sentence Style Correction

## Intended flow of execution
The whole process is divided in several independent steps that are executed seperatly one at the time. A normal ordering of these steps would be :

1. Convert a raw pretraining corpus into a h5 format
2. Generate a train/validation split on the generated h5 dataset
3. Dump the vocabulary of the generated h5 dataset based on the train/valid split
4. Repeat steps 1 and 2 for any finetuning datasets
5. Pretrain the model using the preprocessed pretraining h5 dataset
6. Finetune the model using the preprocessed finetuning h5 dataset
7. Perform inference using the finetuned model

## Execution Modes

### hd5_gen
Mode used to convert a raw corpus into the required h5 format. The **hd5_gen** section in the configurations should be set properly. The source raw corpus should be in a tar.gz archive with the text files at the root. The method was intended for datasets like BookCorpus which are divided in thousands of different files. So, raw datasets **SHOULD NOT** be combined before compressing to tar.gz since each document in the archive is loaded in memory completely. If a document is too large, you will most likely run out of memory. A big document can be split into multiple smaller one before compressing if this is the case. The current implementation assumes the raw corpus has been tokenized and preprocessed. It will split words based on spaces. The code can be adapted so that the preprocessing and/or tokenization is done at this step, but keep in mind that the way it is implemented now, the corpus is processed at a sentence level (a sentence being a single split of the whole corpus on newline characters). For example, BPE could be applied at this step, but the codes must be pre-computed (e.g. using fastBPE).

### gen_split
Mode used to generate a train/valid split for a dataset in the h5 format. The **hd5_gen** section in the configurations should be set properly. Up to 20 splits can be generated on a single h5 dataset. **By default, a new h5 dataset does not contain a split and won't be usable until at least one is generated.**

### vocab_dump
Mode used to dump the vocabulary of a h5 dataset. The **vocab_dump** section in the configurations should be set properly. At least one train/valid split must be available as the vocabulary is only based off the training portion. This vocabulary will be used when using a different dataset than the one used for pretraining. This will generate a vocabulary file in the h5 format also.

### pretrain
Mode used to pretrain a model. The **pretrain**, **optimizer**, **eval** and **transformer_s2s** sections in the configurations should be set properly. A h5 dataset with at least one split generated and the dumped vocabulary are required to start the pretraining. At the interval set in the **eval** section, the model will be evaluated on a random subset of the validation set (based on the number of batches set in the configurations). Only a subset is used so that training does not take too long, but the code can be adapted to process the whole validation set. Before the checkpoints are saved, a random sentence is processed from the validation set and the result is shown on screen with the loss metrics.

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
- Preprocessed finetuning corpus converted to h5
- Vocabulary h5 file used for pretraining
- (optional) source and references text files for GLEU evaluation

Mode used to finetune a pretrained model. The transformer model configurations should be the same as during the pretraining. The finetuning corpus should be preprocessed the same way as the pretraining corpus. If GLEU is used to dertermine the best model, the **gleu** section in the configurations should be set. If the *preprocess* option is set for gleu, the **preprocess** section in the configuration must be set. The preprecessing involves tokenizing with SpaCy, applying BPE using the codes computed with fastBPE for the pretraining and putting everything in lowercase. The GLEU evaluation was intended to be used with the *dev* set of the JFLEG evaluation corpus. Before the checkpoints are saved, a random sentence is processed from the validation set and the result is shown on screen with the loss metrics.

### eval
<ins>Configurations section required</ins> :
- transformer_s2s
- manual_eval

<ins>Files Required</ins> :
- Finetuned model (best or current checkpoint)
- Preprocessed finetuning corpus converted to h5
- Vocabulary h5 file used for pretraining

Mode used to correct sentences manually selected. This mode is intended to test the model quickly during training or after. It will translate all the sentences in the list provided in the configurations and output the result on screen. The sentences should already be preprocessed and tokenized like the pretraing corpus (e.g. Spacy tokenization with BPE applied). 


### inference
<ins>Configurations section required</ins> :
- transformer_s2s
- inference
- (optional) preprocess

<ins>Files Required</ins> :
- Finetuned model (best or current checkpoint)
- Preprocessed finetuning corpus converted to h5
- Vocabulary h5 file used for pretraining
- Text file with source sentences to correct

Mode used to do inference with a finetuned model. It will process each sentence in the source file and output the best decoded sentence on screen and to the specified text file. If the source file is not preprocessed, the *preprocess* option can be set to True and each sentence will be tokenized with SpaCy, BPE will be applied and the text will be put in lowercase. To preprocess the text differently, either change the code or provide a source text file with preprocessed sentences. Following this [paper] (https://www.aclweb.org/anthology/N18-1057.pdf), it is possible to noise the beam decoding scores. This is useful if a reverse model has been trained (dirty to clean) to generate errors that are more diverse. The bigger the *noising_beta* is, the bigger the noise will be. The *temperature* configuration was added to help with the noising since sometimes the best decoding candidate has a score that is a lot better than the other beam candidats, so even with the noise, less probable sentences are not often picked. Basically, the beam decoding scores are divided by the temperature before the softmax is applied. So the bigger the temperature is, the smaller the gap gets between the scores making the noise more effective.
