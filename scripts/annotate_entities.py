# This code is based on the code from the following URL:
# https://github.com/staoxiao/RetroMAE/blob/master/examples/retriever/msmarco/preprocess.py:

import argparse
import os

from datasets import Features, Value, load_from_disk

from kembed.entity_linker.dictionary import DictionaryEntityLinker
from kembed.entity_linker.refined import RefinedEntityLinker

_entity_linker: DictionaryEntityLinker | RefinedEntityLinker | None = None


def _annotate_entities(examples: dict[str, list[str]], args: argparse.Namespace) -> dict[str, list]:
    global _entity_linker

    if _entity_linker is None:
        if args.entity_linker == "dictionary":
            _entity_linker = DictionaryEntityLinker.load(
                data_dir=args.dictionary_path,
                min_link_prob=args.dictionary_min_link_prob,
                min_prior_prob=args.dictionary_min_prior_prob,
                min_link_count=args.dictionary_min_link_count,
            )
        elif args.entity_linker == "refined":
            _entity_linker = RefinedEntityLinker.load(model_name=args.refined_model_name, device=args.refined_device)
        else:
            raise ValueError(f"Invalid entity linker: {args.entity_linker}")

    text_mentions = []
    for mentions in _entity_linker.detect_mentions_batch(examples["text"]):
        text_mentions.append([])
        for mention in mentions:
            text_mentions[-1].append({"kb_id": mention.kb_id, "start": mention.start, "end": mention.end})
    data = {"mentions": text_mentions}

    if "title" in examples:
        title_mentions = []
        for mentions in _entity_linker.detect_mentions_batch(examples["title"]):
            title_mentions.append([])
            for mention in mentions:
                title_mentions[-1].append({"kb_id": mention.kb_id, "start": mention.start, "end": mention.end})
        data["title_mentions"] = title_mentions

    return data


def main(args: argparse.Namespace) -> None:
    dataset = load_from_disk(args.dataset_dir)
    if "title" in dataset.column_names:
        features = Features(
            {
                "id": Value(dtype="string"),
                "text": Value(dtype="string"),
                "title": Value(dtype="string"),
                "mentions": [
                    {"end": Value(dtype="int32"), "kb_id": Value(dtype="string"), "start": Value(dtype="int32")}
                ],
                "title_mentions": [
                    {"end": Value(dtype="int32"), "kb_id": Value(dtype="string"), "start": Value(dtype="int32")}
                ],
            }
        )
    else:
        features = Features(
            {
                "id": Value(dtype="string"),
                "text": Value(dtype="string"),
                "mentions": [
                    {"end": Value(dtype="int32"), "kb_id": Value(dtype="string"), "start": Value(dtype="int32")}
                ],
            }
        )
    dataset = dataset.map(
        _annotate_entities,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.max_workers,
        fn_kwargs={"args": args},
        features=features,
        remove_columns=[name for name in dataset.column_names if name not in features],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to input dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the annotated dataset")
    parser.add_argument(
        "--entity_linker", choices=["dictionary", "refined"], required=True, help="Entity linker to use"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--dictionary_path", type=str, default=None, help="Path to dictionary-based entity linker data")
    parser.add_argument(
        "--dictionary_min_link_prob",
        type=float,
        default=0.05,
        help="Minimum link probability to detect an entity name in the dictionary",
    )
    parser.add_argument(
        "--dictionary_min_prior_prob",
        type=float,
        default=0.3,
        help="Minimum prior probability to detect an entity as a candidate referent",
    )
    parser.add_argument(
        "--dictionary_min_link_count",
        type=int,
        default=1,
        help="Minimum number of incoming links from other articles to detect the entity in the dictionary",
    )
    parser.add_argument(
        "--refined_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for ReFinED linker"
    )
    parser.add_argument(
        "--refined_model_name",
        type=str,
        choices=["aida_model", "questions_model", "wikipedia_model"],
        default="wikipedia_model",
        help="ReFinED model name",
    )

    args = parser.parse_args()

    main(args)
