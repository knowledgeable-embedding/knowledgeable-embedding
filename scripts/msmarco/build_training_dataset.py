import argparse
from collections import defaultdict
from typing import Iterator

from datasets import Dataset


def _read_qrels_file(qrels_file: str) -> dict[str, list[str]]:
    data = defaultdict(list)
    with open(qrels_file) as f:
        for line in f:
            items = line.strip().split("\t")
            data[items[0]].append(items[2])
    return data


def _read_negatives_file(negatives_file: str) -> Iterator[dict[str, str | list[str]]]:
    with open(negatives_file) as f:
        for line in f:
            query_id, negative_passage_ids_str = line.strip().split("\t")
            negative_passage_ids = negative_passage_ids_str.split(",")
            # MEMO: do not sample negatives here
            yield {"query_id": query_id, "negative_passage_ids": negative_passage_ids}


def main(args: argparse.Namespace) -> None:
    qrel_data = _read_qrels_file(args.qrels_file)

    def data_generator():
        for item in _read_negatives_file(args.negatives_file):
            positive_passage_ids = qrel_data[item["query_id"]]
            yield {
                "query_id": item["query_id"],
                "positive_passage_ids": positive_passage_ids,
                "negative_passage_ids": item["negative_passage_ids"],
            }

    dataset = Dataset.from_generator(data_generator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels_file", type=str, required=True, help="Path to qrels TSV file")
    parser.add_argument("--negatives_file", type=str, required=True, help="Path to negatives TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")

    args = parser.parse_args()

    main(args)
