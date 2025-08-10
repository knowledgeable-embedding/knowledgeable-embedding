# This code is based on the code from the following URL:
# https://github.com/staoxiao/RetroMAE/blob/master/examples/retriever/msmarco/preprocess.py:

import argparse
from typing import Iterator

from datasets import Dataset


def _read_query_file(query_file: str) -> Iterator[dict[str, str]]:
    for line in open(query_file):
        items = line.strip("\n").split("\t")
        data = {"id": items[0], "text": items[1]}
        yield data


def main(args: argparse.Namespace) -> None:
    dataset = Dataset.from_generator(_read_query_file, gen_kwargs={"query_file": args.queries_file})
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", type=str, required=True, help="Path to queries TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")

    args = parser.parse_args()

    main(args)
