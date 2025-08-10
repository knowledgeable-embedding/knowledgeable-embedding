import argparse
import csv
from typing import Iterator

from datasets import Dataset


def main(args: argparse.Namespace) -> None:
    def dataset_genarator() -> Iterator[dict[str, str | list[str]]]:
        with open(args.dataset_file) as f:
            for index, row in enumerate(csv.reader(f, delimiter="\t")):
                query = row[0]
                answers = eval(row[1].strip())
                yield {"id": str(index), "text": query, "answers": answers}

    dataset = Dataset.from_generator(dataset_genarator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")

    args = parser.parse_args()

    main(args)
