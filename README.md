## Overview

This repository covers a series of studies on three notable models with various datasets, tokenizers and schdulers.

Models are Sequence-to-Sequence, Attention and Transformer respectively.


Three well-known Machine Translation dataset been used.
Datasets are WMT, IWSLT and Multi30 respectively 

Tokenizer

LR Scheduler


## Directory Hierarchy
```bash
├── checkpoints             # Save Trained model state dicts and Train_record in this directory
├── configs                 # Configuration files for train, model and vocab are saved in this directory
├── data                    # Datasets will be stored in this directory. prepare_data.sh create directory and files within
│   ├── iwslt
│   ├── multi30k
│   └── wmt
├── models                  # Model Structures
│   ├── seq2seq
│   ├── seq2seq_attn
│   └── transformer
├── prepare_data.sh         # This file carries out Download dataset, Build vocab, Tokenize process
├── README.md
├── run.sh                  # Actual Train, Test, Inference with this shell script
├── scripts                 # Contains Shell Scripts for download, build vocab
├── train.py
└── utils                   # Contains Helper functions for data, model, train, test


```



## Getting Started

### Data Preprocessi
First thing to do is prepare datasets.
Original WMT, IWSLT datasets are much greater in volum than multi30k,
and besides for more convinience for using codes on google colab GPU env.
set WMT, IWSLT dataset for 30000 for training, 1000 for validation and test

Of course, using large dataset improves performance.
But the main purpose here is to compare performance of model structures and various techniaues.
Therefore, similar quantities of data will be used.

For all three datasets, the training data set to about 30,000, and the validation * test dataset are set to about 1,000.

Even if there is no physical GPU to use for the experiment, the code realization experiment is tailored to google colab so that you can freely use it in colab.


```bash
bash prepare_data.sh
```

prepare_data.sh downloads datasets, builds vocabs, tokenizes original datasets and make integer mapped sequence based on the generate vocabs.

Tokenizers are Word, BPE, Unigram.
for tokenizing process, sentencepiece library has been used.



### Train Model
the main function 
```bash
bash run.sh -a train -m <model> 
```
