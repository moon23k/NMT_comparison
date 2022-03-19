import os
import math
import yaml
import json
import argparse

import torch
import torch.nn as nn
import sentencepiece as spm

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import seq2seq_valid_epoch, transformer_valid_epoch, create_src_mask, create_trg_mask





class Config(object):
    def __init__(self, args):
        file = f"configs/{args.model}.yaml"

        with open(file, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for p in params.items():
            setattr(self, p[0], p[1])

        self.device = torch.device('cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ': ', value)




def run(args, config):
    #get tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer_path = f"data/{args.data}/vocab/{args.tokenizer}.model"

    #get saved model state dict
    if args.scheduler != "None":
        model_dict = f"checkpoints/{args.model}/{args.data}/{args.tokenizer}/train_states_{args.scheduler}.pt"
    else:
        model_dict = f"checkpoints/{args.model}/{args.data}/{args.tokenizer}/train_states.pt"

    #check if files are existing
    assert os.path.exists(tokenizer_path)
    assert os.path.exists(model_dict)

    tokenizer.load(tokenizer_path)

    #load trained model, criterion
    model = load_model(args.model, config)
    model.load_state_dict(torch.load(model_dict)['model_state_dict'])
    model.eval()


    print('------------------------------------')
    print('Inference Test options')
    print(f'  * model        : {args.model}')
    print(f'  * dataset      : {args.data}')
    print(f'  * tokenizer    : {args.tokenizer}')
    print(f'  * lr_scheduler : {args.scheduler}')
    print('\nIf you wanna stop, type "quit" on user input')

    while True:
        seq = input('\nUser Input sentence >> ')
        if seq == 'quit':
            print('Translator terminated!')
            print('------------------------------------')
            break
        
        #process user input to model src tensor
        src = tokenizer.EncodeAsIds(seq)
        src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
        src_mask = create_src_mask(src)

        src = model.embedding(src)

        with torch.no_grad():
            enc_out = model.encoder(src, src_mask)
        
        trg_indice = [tokenizer.bos_id()]


        while True:
            trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
            trg_mask = create_trg_mask(trg_tensor)

            trg = model.embedding(trg)

            with torch.no_grad():
                dec_out, _ = model.decoder()
                out = model.fc_out(dec_out)

            pred_token = out.argmax(2)[:, -1].item()
            trg_indice.append(pred_token)

            if pred_token == tokenizer.eos_id():
                break
        
        pred_seq = trg_indice[1:]
        pred_seq = tokenizer.Decode(pred_seq)

        print(f"Translated sentence >> {pred_seq}")




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
