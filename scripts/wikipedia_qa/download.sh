#!/bin/bash

WIKIPEDIA_QA_DATA_DIR=${WIKIPEDIA_QA_DATA_DIR:-"data/wikipedia_qa"}

mkdir -p ${WIKIPEDIA_QA_DATA_DIR}
cd ${WIKIPEDIA_QA_DATA_DIR}

curl https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz | gunzip -c > psgs_w100.tsv
curl https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz | gunzip -c > biencoder-nq-train.json
curl https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz | gunzip -c > biencoder-trivia-train.json
curl https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-train.json.gz | gunzip -c > biencoder-webquestions-train.json
curl https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz | gunzip -c > biencoder-curatedtrec-train.json

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
curl https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz | gunzip -c > trivia-test.qa.csv

mkdir -p entity_questions
cd entity_questions

wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip

unzip dataset.zip
mv dataset/* .
rm -rf dataset dataset.zip
