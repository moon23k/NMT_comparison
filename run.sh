#!/bin/bash
python3 -m pip install -U PyYAML

while getopts a:m:d:t:s: flag; do
    case "${flag}" in
        a) action=${OPTARG};;
        m) model=${OPTARG};;
        d) data=${OPTARG};;
        t) tokenizer=${OPTARG};;
        s_ scheduler=${OPTARG};;
    esac
done


echo "$action | model: $model / tokenizer: $tokenizer / scheduler: $s"

if [ $action='train' ]; then
    python3 train.py -model $model -data $data -tokenizer $tokenizer -scheduler $scheduler


elif [ $action='test' ]; then
    python3 test.py -model $model -data $data -tokenizer $tokenizer -scheduler $scheduler

elif [ $action='inference' ]; then
    python3 inference.py -model $model -data $data -tokenizer $tokenizer -scheduler $scheduler
fi