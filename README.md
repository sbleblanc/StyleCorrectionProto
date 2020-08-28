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