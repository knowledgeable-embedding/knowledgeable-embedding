# This code is based on the code from the following URL:
# https://github.com/staoxiao/RetroMAE/blob/master/examples/retriever/msmarco/preprocess.py:

import argparse
from typing import Iterator

from datasets import Dataset


def _read_corpus_file(corpus_file: str) -> Iterator[dict[str, str]]:
    for line in open(corpus_file):
        items = line.strip("\n").split("\t")
        data = {"id": items[0], "title": items[1], "text": items[2]}
        yield data


def main(args: argparse.Namespace) -> None:
    dataset = Dataset.from_generator(_read_corpus_file, gen_kwargs={"corpus_file": args.corpus_file})
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to corpus TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")

    args = parser.parse_args()

    main(args)
