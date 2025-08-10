import argparse
import json
import os
from typing import Iterator

from datasets import Dataset


def main(args: argparse.Namespace) -> None:
    def dataset_genarator() -> Iterator[dict[str, str]]:
        with open(args.dataset_file) as f:
            examples = json.load(f)
            dataset_name = os.path.basename(args.dataset_file).replace(".json", "")
            for example_id, example in enumerate(examples):
                query_id = f"{dataset_name}_{example_id}"
                yield {"id": query_id, "text": example["question"], "answers": example["answers"]}

    dataset = Dataset.from_generator(dataset_genarator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")

    args = parser.parse_args()

    main(args)
