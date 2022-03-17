#!/bin/bash
mkdir -p data
cd data


datasets=(multi30k iwslt wmt)
splits=(train valid test)
tokenizers=(word bpe unigram)
langs=(en de)



#Download datasets
for dataset in "${datasets[@]}"; do
    bash ../scripts/download_${dataset}.sh
done



#Peer Tokenize with sacremoses
python3 -m pip install -U sacremoses
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for lang in "${langs[@]}"; do
            mkdir -p ${dataset}/tok
            sacremoses -l ${lang} -j 8 tokenize < ${dataset}/raw/${split}.${lang} > ${dataset}/tok/${split}.${lang}
        done
    done
done



#get sentencepiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../



#concat all files into one file to build vocab
for dataset in "${datasets[@]}"; do
    cat ${dataset}/tok/* >> ${dataset}/concat.txt
done



#Build Vocab
for dataset in "${datasets[@]}"; do
    mkdir -p ${dataset}/vocab
    for tokenizer in "${tokenizers[@]}"; do
        bash ../scripts/build_vocab.sh -i ${dataset}/concat.txt -p ${dataset}/vocab/${tokenizer} -t ${tokenizer}
    done
    rm ${dataset}/concat.txt
done



#tokens to ids | total 54 files
for dataset in "${datasets[@]}"; do
    mkdir -p ${dataset}/ids
    for split in "${splits[@]}"; do
        for tokenizer in "${tokenizers[@]}"; do
            for lang in "${langs[@]}"; do
                mkdir -p ${dataset}/ids/${tokenizer}
                spm_encode --model=${dataset}/vocab/${tokenizer}.model --extra_options=bos:eos \
                --output_format=id < ${dataset}/tok/${split}.${lang} > ${dataset}/ids/${tokenizer}/${split}.${lang}
            done
        done
    done
done


#remove sentencepiece
rm -r sentencepiece