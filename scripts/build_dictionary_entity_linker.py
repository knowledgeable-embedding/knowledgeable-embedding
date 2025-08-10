import argparse
import json
import math
import multiprocessing
import os
import re
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial

import datasets
import numpy as np
import spacy
from marisa_trie import Trie
from tqdm import tqdm

from kembed.entity_linker.dictionary import NONE_ID, get_tokenizer
from kembed.utils import load_tsv_mapping, normalize_wikipedia_title

DISAMBI_MATCHER = re.compile(r"\s\(.*\)$")

_name_trie: Trie
_tokenizer: spacy.tokenizer.Tokenizer
_max_mention_length: int


def _init_name_occurrence_worker(name_trie: Trie, max_mention_length: int):
    global _name_trie, _tokenizer, _max_mention_length
    _name_trie = name_trie
    _max_mention_length = max_mention_length
    _tokenizer = get_tokenizer("en")


def _extract_name_occurrences(examples: dict[str, list[str]], case_sensitive: bool) -> list[str]:
    ret = []
    for doc_title, doc_text in zip(examples["title"], examples["text"]):
        doc_text = "\n".join([doc_title, doc_text])
        for paragraph_text in doc_text.split("\n"):  # some tokenizers have trouble with long texts
            tokens = _tokenizer(paragraph_text)
            if not case_sensitive:
                paragraph_text = paragraph_text.lower()

            end_offsets = frozenset(token.idx + len(token) for token in tokens)
            names = []
            for token in tokens:
                for prefix in _name_trie.prefixes(paragraph_text[token.idx : token.idx + _max_mention_length]):
                    if token.idx + len(prefix) in end_offsets:
                        names.append(prefix)

            ret += list(frozenset(names))

    return ret


def main(args: argparse.Namespace) -> None:
    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)

    name_wikidata_id_counter = defaultdict(Counter)
    with tqdm(total=math.ceil(len(dataset) / args.batch_size)) as pbar:
        for examples in dataset.iter(batch_size=args.batch_size):
            for title in examples["title"]:
                title = normalize_wikipedia_title(title)
                wikidata_id = wikidata_id_mapping.get(title, NONE_ID)

                name = DISAMBI_MATCHER.sub("", title)
                if not args.case_sensitive:
                    name = name.lower()
                name_wikidata_id_counter[name][wikidata_id] += 1

            for text, mentions in zip(examples["text"], examples["mentions"]):
                for mention in mentions:
                    if mention["source"] == "bold_text":
                        continue

                    name = text[mention["start"] : mention["end"]]
                    if not args.case_sensitive:
                        name = name.lower()

                    if len(name) <= args.max_mention_length:
                        wikidata_id = mention["kb_id"]
                        if wikidata_id is None:
                            wikidata_id = NONE_ID
                        name_wikidata_id_counter[name][wikidata_id] += 1

            pbar.update()

    all_name_trie = Trie(name_wikidata_id_counter.keys())
    name_counter = Counter()
    with closing(
        multiprocessing.Pool(
            processes=args.max_workers,
            initializer=_init_name_occurrence_worker,
            initargs=(all_name_trie, args.max_mention_length),
            maxtasksperchild=args.max_tasks_per_worker,
        )
    ) as pool:
        del all_name_trie
        with tqdm(total=math.ceil(len(dataset) / args.batch_size)) as pbar:
            func = partial(_extract_name_occurrences, case_sensitive=args.case_sensitive)
            for names in pool.imap_unordered(
                func, dataset.select_columns(["title", "text"]).iter(batch_size=args.batch_size)
            ):
                name_counter.update(names)
                pbar.update()

    wikidata_ids = list(
        frozenset(
            wikidata_id
            for wikidata_id_counter in name_wikidata_id_counter.values()
            for wikidata_id in wikidata_id_counter.keys()
        )
    )
    wikidata_id_trie = Trie(wikidata_ids)
    name_data_dict = defaultdict(list)

    for name, wikidata_id_counter in name_wikidata_id_counter.items():
        doc_count = name_counter[name]
        total_link_count = sum(wikidata_id_counter.values())

        if doc_count == 0:
            continue

        link_prob = total_link_count / doc_count
        if link_prob < args.min_link_prob:
            continue

        for wikidata_id, link_count in wikidata_id_counter.items():
            if link_count < args.min_link_count:
                continue

            prior_prob = link_count / total_link_count
            if prior_prob < args.min_prior_prob:
                continue

            name_data_dict[name].append((wikidata_id_trie[wikidata_id], link_count, total_link_count, doc_count))

    del name_counter, name_wikidata_id_counter

    name_trie = Trie(name_data_dict.keys())
    name_data = []
    offsets = []
    for name in sorted(name_trie, key=lambda n: name_trie[n]):
        offsets.append(len(name_data))
        name_data += name_data_dict[name]
    offsets.append(len(name_data))

    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "data.npy"), np.array(name_data, dtype=np.uint32))
    del name_data
    np.save(os.path.join(args.output_dir, "offsets.npy"), np.array(offsets, dtype=np.uint32))
    del offsets

    wikidata_id_trie.save(os.path.join(args.output_dir, "wikidata_id.trie"))
    name_trie.save(os.path.join(args.output_dir, "name.trie"))

    with open(os.path.join(args.output_dir, "config.json"), "w") as config_file:
        json.dump(
            {
                "max_mention_length": args.max_mention_length,
                "case_sensitive": args.case_sensitive,
                "min_link_prob": args.min_link_prob,
                "min_prior_prob": args.min_prior_prob,
                "min_link_count": args.min_link_count,
            },
            config_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir", type=str, required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument("--wikidata_id_file", type=str, required=True, help="Path to Wikidata ID mapping TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dictionary data")
    parser.add_argument(
        "--max_mention_length", type=int, default=100, help="Maximum length of entity mention to detect"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument("--max_tasks_per_worker", type=int, default=16, help="Maximum tasks per worker process")
    parser.add_argument(
        "--min_link_prob", type=float, default=0.05, help="Minimum link probability to detect an entity name"
    )
    parser.add_argument(
        "--min_prior_prob",
        type=float,
        default=0.3,
        help="Minimum prior probability to detect an entity as a candidate referent",
    )
    parser.add_argument(
        "--min_link_count",
        type=float,
        default=1,
        help="Minimum number of incoming links from other articles to detect the entity",
    )
    parser.add_argument("--case_sensitive", action="store_true", help="Enable case-sensitive matching for entity names")

    args = parser.parse_args()

    main(args)
