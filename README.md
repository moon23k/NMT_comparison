## Overview
> **_This repository covers a series of studies on three notable models with various datasets, tokenizers and schdulers._**
<br>

NLP has various sub-tasks, and Machine Translation(MT) is one of the representative one.  
MT has been developed into Neural Machine Translation(NMT) through Rule-Based, Statistical Machine Translation.

In this repository, three notable models of NMT are implemented in pytorch code.  
Models are **Sequence to Sequence**, **Attention**, and **Transformer** respectively.

Experimental variables were prepared to measure the performance of the model in various ways.  
Dataset, Tokenizer and LR Scheduler are the prepared variables. Each of them has sub variables as follows.

<br>
<br>

## Models

#### Sequence to Sequence &nbsp; | &nbsp; [paper](https://arxiv.org/abs/1409.3215)
Before sequence to sequence model was devised, NMT was used as a secondary tool to make better use of SMT. The main reason was that the relationship between elements in a language sequence could not be properly grasped with a linear model structure of the RNN cells.

The cited paper suggested that the reverse method was used to improve performance, but that part was omitted from the code implementation.

<br>

#### Attention Mechanism &nbsp; | &nbsp; [paper](https://arxiv.org/abs/1409.0473)



<br>

#### Transformer &nbsp; | &nbsp; [paper](https://arxiv.org/abs/1706.03762)



<br>
<br>


## Experimental Variables

#### Data

* WMT_SM


* IWSTL_SM


* Multi30k



<br>

#### Tokenizer

* Word

* BPE

* Unigram



<br>

#### LR Scheduler

* None

* Cosine Annealing

* Cosine Annealing with Warm-up

* Exponential

* Step


<br>
<br>


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

<br>



## Getting Started

### Data Preprocessig
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


<br>

### Train Model
the main function 
```bash
bash run.sh -a train -m <model> 
```
