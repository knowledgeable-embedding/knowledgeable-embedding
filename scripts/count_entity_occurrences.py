import argparse
import math
from collections import Counter

import datasets
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    counter = Counter()

    with tqdm(total=math.ceil(len(dataset) / args.batch_size)) as pbar:
        for examples in dataset.iter(batch_size=args.batch_size):
            for mentions in examples["mentions"]:
                for mention in mentions:
                    if mention["source"] == "bold_text":
                        continue

                    kb_id = mention["kb_id"]
                    if kb_id is not None:
                        counter[kb_id] += 1

            pbar.update()

    with open(args.output_file, "w") as f:
        for kb_id, count in counter.items():
            f.write(f"{kb_id}\t{count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir", type=str, required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    main(args)
