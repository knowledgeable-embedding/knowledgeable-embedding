import argparse
import bz2
import json
import logging
import multiprocessing
import os
from contextlib import closing

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    languages = args.languages.split(",")
    language_set = frozenset(languages)

    output_files = {
        language: open(os.path.join(args.output_dir, f"{language}-wikidata-ids.tsv"), "w") for language in languages
    }

    with closing(
        multiprocessing.Pool(
            processes=args.max_workers,
            initializer=_init_worker,
            initargs=(language_set,),
        )
    ) as pool:
        with bz2.open(args.dump_file, "rt") as input_file:
            counter = 0
            for result in pool.imap(_process_line, input_file, chunksize=args.chunk_size):
                for language, title, wikidata_id in result:
                    output_files[language].write(f"{title}\t{wikidata_id}\n")
                counter += 1
                if counter % 100000 == 0 and counter != 0:
                    logger.info(f"Processed {counter} lines")

    for output_file in output_files.values():
        output_file.close()


_language_set: frozenset[str] | None


def _init_worker(language_set: frozenset[str] | None) -> None:
    global _language_set
    _language_set = language_set


def _process_line(line: str) -> list[tuple[str, str, str]]:
    line = line.rstrip()
    if line in ("[", "]"):
        return []

    if line[-1] == ",":
        line = line[:-1]
    item = json.loads(line)
    if item["type"] != "item":
        return []

    ret = []
    wikidata_id = item["id"]
    for link_item in item["sitelinks"].values():
        site = link_item["site"]
        if not site.endswith("wiki"):
            continue
        language = site[:-4]
        if _language_set and language not in _language_set:
            continue

        ret.append((language, link_item["title"], wikidata_id))

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True, help="Path to Wikidata JSON dump in bz2 format")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for TSV files")
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument(
        "--max_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of lines processed per chunk")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
