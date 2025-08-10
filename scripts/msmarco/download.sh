#!/bin/bash
# Copied from https://github.com/staoxiao/RetroMAE/blob/master/examples/retriever/msmarco/get_data.sh

MSMARCO_DATA_DIR=${MSMARCO_DATA_DIR:-"data/msmarco"}

mkdir -p ${MSMARCO_DATA_DIR}
cd ${MS_MARCO_DATA_DIR}
wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
mv marco/* .
rm -rf marco

curl https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz | gunzip -c > qidpidtriples.train.full.2.tsv
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv

join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train_negs.tsv