import argparse
import logging

import datasets

from kembed.utils import ReverseIndexTrie

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    entity_reverse_index = ReverseIndexTrie.load(args.entity_reverse_index_file)
    entities = set()

    if args.preprocessed_dataset_dir:
        all_datasets = []
        for dataset_dir in args.preprocessed_dataset_dir:
            dataset = datasets.load_from_disk(dataset_dir)
            all_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(all_datasets)
        dataset.select_columns("mentions")

        for examples in dataset.iter(batch_size=args.batch_size):
            for mentions in examples["mentions"]:
                for mention in mentions:
                    kb_id = mention["kb_id"]
                    if kb_id is None:
                        continue
                    if kb_id in entities:
                        continue
                    if len(entity_reverse_index[kb_id]) < args.min_count:
                        continue

                    entities.add(kb_id)

    else:
        for kb_id, entries in entity_reverse_index.items():
            if len(entries) >= args.min_count:
                entities.add(kb_id)

    with open(args.output_file, "w") as f:
        f.write("[PAD]\t0\n")
        for index, kb_id in enumerate(entities, 1):
            f.write(f"{kb_id}\t{index}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir",
        type=str,
        nargs="*",
        help="Paths to one or more preprocessed dataset directories (space-separated)",
    )
    parser.add_argument(
        "--entity_reverse_index_file", type=str, required=True, help="Path to the entity reverse index file"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output entity vocabulary file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--min_count", type=int, default=1, help="Minimum occurrence count to include an entity in the vocabulary"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
