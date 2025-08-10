import argparse
import json
import os
from typing import Iterator

from datasets import Dataset


def main(args: argparse.Namespace) -> None:
    def dataset_generator() -> Iterator[dict[str, str | list[str]]]:
        with open(args.dataset_file) as f:
            examples = json.load(f)
            dataset_name = os.path.basename(args.dataset_file).replace(".json", "")
            passage_key = None
            for example_id, example in enumerate(examples):
                if not example["positive_ctxs"] or not example["hard_negative_ctxs"]:
                    continue

                if passage_key is None:
                    if "passage_id" in example["positive_ctxs"][0]:
                        passage_key = "passage_id"
                    else:
                        passage_key = "psg_id"

                query_id = f"{dataset_name}_{example_id}"

                if args.top_positive_only:
                    positive_passage_ids = [example["positive_ctxs"][0][passage_key]]
                else:
                    positive_passage_ids = [o[passage_key] for o in example["positive_ctxs"]]

                negative_passage_ids = [o[passage_key] for o in example["hard_negative_ctxs"]]

                yield {
                    "query_id": query_id,
                    "positive_passage_ids": positive_passage_ids,
                    "negative_passage_ids": negative_passage_ids,
                }

    dataset = Dataset.from_generator(dataset_generator)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--top_positive_only", action="store_true", help="Use only the top positive passage")

    args = parser.parse_args()

    main(args)
