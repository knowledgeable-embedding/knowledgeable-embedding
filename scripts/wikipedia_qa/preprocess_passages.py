import argparse
import csv
from typing import Iterator

from datasets import Dataset


def main(args: argparse.Namespace) -> None:
    def dataset_genarator() -> Iterator[dict[str, str]]:
        with open(args.passages_file) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                yield {"id": row["id"], "text": row["text"], "title": row["title"]}

    dataset = Dataset.from_generator(dataset_genarator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
