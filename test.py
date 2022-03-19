import os
import math
import yaml
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import seq2seq_valid_epoch, transformer_valid_epoch




class Config(object):
    def __init__(self, args):
        files = [f"configs/{file}.yaml" for file in [args.model, 'test']]

        for file in files:
            with open(file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            for p in params.items():
                setattr(self, p[0], p[1])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ': ', value)




def run(args, config):
    #get saved model state dict    
    if args.scheduler != "None":
        model_dict = f"checkpoints/{args.model}/{args.data}/train_states_{args.scheduler}.pt"
    else:
        model_dict = f"checkpoints/{args.model}/{args.data}/train_states.pt"


    #get dataloader from chosen dataset
    test_dataloader = get_dataloader(args.data, args.tokenizer, 'test', config.batch_size)


    #load trained model, criterion
    model = load_model(args.model, config)
    model.load_state_dict(torch.load(model_dict)['model_state_dict'])
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)
    

    if args.model == 'transformer':
        test_loss = transformer_valid_epoch(model, test_dataloader, criterion, config.device)
    else:
        test_loss = seq2seq_valid_epoch(model, test_dataloader, criterion, config.device)
    
    test_ppl = math.exp(test_loss)
    print(f" Test Loss: {test_loss:.3f} | Test PPL : {round(test_ppl, 2)}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-data', required=True)
    parser.add_argument('-tokenizer', required=True)
    parser.add_argument('-scheduler', required=False)
    args = parser.parse_args()
    
    assert args.model in ['seq2seq', 'seq2seq_attn', 'transformer']
    assert args.data in ['wmt', 'iwslt', 'multi30k']
    assert args.tokenizer in ['word', 'bpe', 'unigram']
    assert args.scheduler in ["None", 'cosine_annealing_warm', 'cosine_annealing', 'exponential', 'step']
    
    config = Config(args)
    
    run(args, config)
