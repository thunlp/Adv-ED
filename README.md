# Adv-ED
Source code and dataset for NAACL 2019 paper "Adversarial Training for Weakly Supervised Event Detection".

## Requirements

- python == 3.6.3
- pytorch == 0.4.1
- numpy == 1.15.2
- sklearn == 0.20.0
- pytorch-pretrained-bert == 0.2.0

## Data

Due to the licence issues, we cannot share the source ACE2005 dataset or the preprocessed data.

So we specify the data format in `DataFormat.md` and you can preprocess the data follow the format.

## Run

Put the preprocessed `.npy` data files in the same directory as the codes.

For the BERT models, download the `Bert_base_uncase` model in `../../BERT_CACHE`.

Run `python train.py` in corresponding directory to train the model.

If you want to tune the hyper parameters, see the `constant.py` and change the parameters defined in the file.

## Cite

If the codes help you, please cite the following paper:

**Adversarial Training for Weakly Supervised Event Detection.** _Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, Peng Li._ NAACL-HLT 2019.
