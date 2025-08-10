import argparse
import multiprocessing
import os

import transformers
from datasets import Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kembed.entity_linker.mention import BaseMention
from kembed.utils import load_tsv_mapping
from kembed.utils.preprocessing import preprocess_text

_tokenizer: PreTrainedTokenizerBase
_entity_vocab: dict[str, int]

transformers.logging.set_verbosity_error()


def _process_example(example: dict, no_title: bool) -> dict[str, list[int]]:
    text = example["text"]
    mentions = []
    if "mentions" in example:
        mentions = [
            BaseMention(kb_id=m["kb_id"], text=text[m["start"] : m["end"]], start=m["start"], end=m["end"])
            for m in example["mentions"]
        ]

    title = None
    title_mentions = None
    if not no_title:
        if "title" in example:
            title = example["title"]
            title_mentions = [
                BaseMention(kb_id=m["kb_id"], text=title[m["start"] : m["end"]], start=m["start"], end=m["end"])
                for m in example.get("title_mentions", [])
            ]

    ret = preprocess_text(text, mentions, title, title_mentions, _tokenizer, _entity_vocab)
    ret["id"] = example["id"]
    return ret


def main(args: argparse.Namespace) -> None:
    global _tokenizer, _entity_vocab
    _tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    os.makedirs(args.output_dir, exist_ok=True)

    _entity_vocab = {}
    if args.entity_vocab_file:
        _entity_vocab = load_tsv_mapping(args.entity_vocab_file, int)

    dataset = load_from_disk(args.preprocessed_dataset_dir)
    dataset = dataset.map(
        _process_example,
        remove_columns=dataset.column_names,
        num_proc=args.max_workers,
        fn_kwargs={"no_title": args.no_title},
        features=Features(
            {
                "id": Value(dtype="string"),
                "input_ids": Sequence(feature=Value(dtype="int32")),
                "entity_ids": Sequence(feature=Value(dtype="int32")),
                "entity_start_positions": Sequence(feature=Value(dtype="int32")),
                "entity_lengths": Sequence(feature=Value(dtype="int32")),
            }
        ),
    )
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir", type=str, required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the processed dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--entity_vocab_file", type=str, help="Path to entity vocabulary TSV file")
    parser.add_argument(
        "--max_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument("--no_title", action="store_true", default=False, help="Exclude title from processing")

    args = parser.parse_args()

    main(args)
