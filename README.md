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
Mode used to convert a raw corpus into the required h5 format. The **hd5_gen** section in the configurations should be set properly. The source raw corpus should be in a tar.gz archive with the text files at the root. The method was intended for datasets like BookCorpus which are divided ine thousands of different files. So, raw datasets **SHOULD NOT** be combined before compressing to tar.gz since each document in the archive is loaded in memory completely. If a document is too large, you will run out of RAM. A big document can be split into multiple smaller one before compressing if this is the case.

### gen_split
Mode used to generate a train/valid split for a dataset in the h5 format. The **hd5_gen** section in the configurations should be set properly. Up to 20 splits can be generated on a single h5 dataset. **By default, a new h5 dataset does not contain a split and won't be usable until at least one is generated.**

### vocab_dump
Mode used to dump the vocabulary of a h5 dataset. The **vocab_dump** section in the configurations should be set properly. At least one train/valid split must be available as the vocabulary is only based off the training portion. This vocabulary will be used when using a different dataset than the one used for pretraining.

### pretrain


### finetune


### eval


### inference

