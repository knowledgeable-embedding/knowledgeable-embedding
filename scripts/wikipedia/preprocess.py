import argparse
import bz2
import logging
import multiprocessing
import xml.etree.ElementTree as ET
from typing import Iterator, NamedTuple
from xml.etree.ElementTree import iterparse

import datasets
import mwparserfromhell

from kembed.utils import load_tsv_mapping, normalize_wikipedia_title

logger = logging.getLogger(__name__)

_redirects: dict[str, str]
_wikidata_id_mapping: dict[str, str]


class WikiLink(NamedTuple):
    title: str
    text: str


class BoldText(NamedTuple):
    text: str


def _extract_pages(dump_file: str) -> Iterator[dict[str, str]]:
    with bz2.open(dump_file, "rt", encoding="utf-8") as f:
        context = iterparse(f, events=("end",))
        for _, elem in context:
            if not elem.tag.endswith("page"):
                continue
            namespace = elem.tag[:-4]

            title = elem.find(f"./{namespace}title").text
            ns = elem.find(f"./{namespace}ns").text
            redirect = elem.find(f"./{namespace}redirect")
            if redirect is not None:
                continue

            # filter pages that are not in the "main" namespace.
            if ns != "0":
                elem.clear()
                continue

            text = elem.find(f"./{namespace}revision/{namespace}text").text or ""
            elem.clear()

            yield {"title": title, "wiki_text": text}


def _parse_articles(
    examples: dict[str, list[str]],
    treat_first_bold_text_as_link: bool,
    max_chars_before_bold_text: int,
) -> dict[str, list[str] | list[list[dict]]]:
    texts = []
    mentions_list: list[list[dict]] = []

    for title, wiki_text in zip(examples["title"], examples["wiki_text"]):
        try:
            parsed_wiki_text = mwparserfromhell.parse(wiki_text)
        except Exception:
            logger.warning(f"Failed to parse wiki text: {title}")
            texts.append("")
            mentions_list.append([])
            continue

        detected_first_bold_text = False
        text = ""
        mentions: list[dict] = []

        for section_index, section in enumerate(
            parsed_wiki_text.get_sections(flat=True, include_lead=True, include_headings=True)
        ):
            for node in section.ifilter(recursive=False):
                for parsed_node in _parse_node(node):
                    if isinstance(parsed_node, WikiLink):
                        mention_title = parsed_node.title

                        if mention_title and "#" not in mention_title:
                            mention_title = _get_normalized_title(mention_title)
                            if mention_title is not None:
                                mention_title = _redirects.get(mention_title, mention_title)
                                kb_id = _wikidata_id_mapping.get(mention_title, None)
                                start = len(text)
                                end = start + len(parsed_node.text)
                                mentions.append(
                                    {
                                        "start": start,
                                        "end": end,
                                        "title": mention_title,
                                        "kb_id": kb_id,
                                        "source": "link",
                                    }
                                )
                        text += parsed_node.text

                    elif isinstance(parsed_node, BoldText):
                        if (
                            treat_first_bold_text_as_link
                            and section_index == 0
                            and len(text.replace("\n", "")) < max_chars_before_bold_text
                            and not detected_first_bold_text
                            and parsed_node.text
                        ):
                            kb_id = _wikidata_id_mapping.get(title, None)
                            start = len(text)
                            end = start + len(parsed_node.text)
                            mentions.append(
                                {
                                    "start": start,
                                    "end": end,
                                    "title": title,
                                    "kb_id": kb_id,
                                    "source": "bold_text",
                                }
                            )
                            detected_first_bold_text = True

                        text += parsed_node.text

                    else:  # str
                        text += parsed_node

        texts.append(text)
        mentions_list.append(mentions)

    return {"text": texts, "mentions": mentions_list}


def _parse_node(node: mwparserfromhell.nodes.Node) -> list[str | BoldText | WikiLink]:
    if isinstance(node, mwparserfromhell.nodes.Wikilink):
        lowered_title = str(node.title).lower()
        if (
            lowered_title.startswith("file:")
            or lowered_title.startswith("image:")
            or lowered_title.startswith("media:")
        ):
            return []

        text = _extract_text_from_node(node)
        if lowered_title.startswith("category:") and text.lower().startswith("category:"):
            return []

        title = node.title.strip_code().strip(" ")
        return [WikiLink(text=text, title=title)]

    elif isinstance(node, mwparserfromhell.nodes.Tag):
        # ignore references and tables
        if str(node.tag) in {"ref", "table"}:
            return []
        elif str(node.tag) == "b":
            return [BoldText(text=_extract_text_from_node(node))]
        elif not mwparserfromhell.definitions.is_visible(node.tag):
            return []

        parsed_child_nodes = []
        for child_node in node.contents.ifilter(recursive=False):
            parsed_child_nodes += _parse_node(child_node)
        return parsed_child_nodes

    text = _extract_text_from_node(node)
    return [text]


def _get_normalized_title(title: str) -> str | None:
    if title.startswith(f"https://en.wikipedia.org/wiki/"):
        title = title[len(f"https://en.wikipedia.org/wiki/") :]
    elif title.startswith("http://") or title.startswith("https://"):
        return None
    elif title.startswith("/wiki/"):
        title = title[len("/wiki/") :]
    elif title.startswith(f"w:en:"):
        title = title[len(f"w:en:") :]
    elif title.startswith("w:") and ":" not in title[len("w:") :]:
        title = title[len("w:") :]
    elif title.startswith(f":en:"):
        title = title[len(f":en:") :]
    elif title.startswith(":"):
        title = title[1:]

    if not title:
        return None

    title = normalize_wikipedia_title(title)
    return title


def _extract_text_from_node(node: mwparserfromhell.nodes.Node) -> str:
    text = node.__strip__(normalize=True, collapse=True, keep_template_params=False)
    if text is None:
        return ""

    return str(text)


def main(args: argparse.Namespace) -> None:
    global _redirects, _wikidata_id_mapping

    _redirects = load_tsv_mapping(args.redirect_file)
    _wikidata_id_mapping = load_tsv_mapping(args.wikidata_id_file)

    dataset = datasets.Dataset.from_generator(_extract_pages, gen_kwargs={"dump_file": args.dump_file})
    dataset = dataset.map(
        _parse_articles,
        batched=True,
        remove_columns=["wiki_text"],
        num_proc=args.max_workers,
        fn_kwargs={
            "treat_first_bold_text_as_link": args.treat_first_bold_text_as_link,
            "max_chars_before_bold_text": args.max_chars_before_bold_text,
        },
    )
    dataset = dataset.filter(
        lambda x: x["text"] != "" and len(x["mentions"]) > 0,
    )
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True, help="Path to Wikipedia XML dump in bz2 format")
    parser.add_argument("--redirect_file", type=str, required=True, help="Path to Wikipedia redirects TSV file")
    parser.add_argument("--wikidata_id_file", type=str, required=True, help="Path to Wikidata ID mapping TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument(
        "--max_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument(
        "--treat_first_bold_text_as_link",
        action="store_true",
        help="Treat the first bold text as a link to the page itself",
    )
    parser.add_argument(
        "--max_chars_before_bold_text", type=int, default=50, help="Max chars before bold text to treat as link"
    )

    args = parser.parse_args()
    main(args)
