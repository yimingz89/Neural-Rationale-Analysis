#!/usr/bin/env bash

echo "-- SST --"
mkdir -p data/sst1
cd data/sst1

echo "Downloading SST data"
wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip -j trainDevTestTrees_PTB.zip 

echo "Downloading filtered word embeddings"
wget https://gist.github.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt

