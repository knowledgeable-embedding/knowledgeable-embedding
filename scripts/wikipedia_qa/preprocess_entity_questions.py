import argparse
import json
import os
from typing import Iterator

from datasets import Dataset


def main(args: argparse.Namespace) -> None:
    def dataset_genarator() -> Iterator[dict[str, str | list[str]]]:
        for dataset_file in os.listdir(args.dataset_dir):
            if not dataset_file.endswith(".json"):
                continue
            with open(os.path.join(args.dataset_dir, dataset_file)) as f:
                examples = json.load(f)
                for example_id, example in enumerate(examples):
                    subset = dataset_file.split(".")[0]
                    query_id = f"{subset}_{example_id}"
                    yield {"id": query_id, "text": example["question"], "answers": example["answers"], "subset": subset}

    dataset = Dataset.from_generator(dataset_genarator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to directory containing dataset JSON files"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")

    args = parser.parse_args()

    main(args)
