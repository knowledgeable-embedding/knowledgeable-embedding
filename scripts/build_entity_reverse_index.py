import argparse
import logging
import math

import datasets
from tqdm import tqdm

from kembed.utils import ReverseIndexEntry, ReverseIndexTrie

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    dataset.select_columns("mentions")

    logger.info("Building entity reverse index...")
    kb_ids = []
    values = []
    with tqdm(total=math.ceil(len(dataset) / args.batch_size)) as pbar:
        for i, examples in enumerate(dataset.iter(batch_size=args.batch_size)):
            for j, mentions in enumerate(examples["mentions"]):
                index = i * args.batch_size + j
                for k, mention in enumerate(mentions):
                    if mention["kb_id"] is not None:
                        kb_ids.append(mention["kb_id"])
                        values.append(ReverseIndexEntry(index, k))

            pbar.update()

    logger.info("Saving entity index...")
    entity_index = ReverseIndexTrie.build(kb_ids, values)
    entity_index.save(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir", type=str, required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to output entity reverse index file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
