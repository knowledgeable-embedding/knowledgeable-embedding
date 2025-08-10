import json
import os
from dataclasses import dataclass

import numpy as np
import spacy
from marisa_trie import Trie

from .mention import BaseMention

NONE_ID = "<None>"


def get_tokenizer(language: str) -> spacy.tokenizer.Tokenizer:
    language_obj = spacy.blank(language)
    return language_obj.tokenizer


@dataclass
class DictionaryELMention(BaseMention):
    link_count: int
    total_link_count: int
    doc_count: int

    @property
    def link_prob(self) -> float:
        if self.doc_count > 0:
            return min(1.0, self.total_link_count / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self) -> float:
        if self.total_link_count > 0:
            return min(1.0, self.link_count / self.total_link_count)
        else:
            return 0.0


class DictionaryEntityLinker:
    def __init__(
        self,
        name_trie: Trie,
        kb_id_trie: Trie,
        data: np.ndarray,
        offsets: np.ndarray,
        max_mention_length: int,
        case_sensitive: bool,
        min_link_prob: float,
        min_prior_prob: float,
        min_link_count: int,
    ):
        self._name_trie = name_trie
        self._kb_id_trie = kb_id_trie
        self._data = data
        self._offsets = offsets
        self._max_mention_length = max_mention_length
        self._case_sensitive = case_sensitive
        self._min_link_prob = min_link_prob
        self._min_prior_prob = min_prior_prob
        self._min_link_count = min_link_count

        self._tokenizer = get_tokenizer("en")

    @staticmethod
    def load(
        data_dir: str,
        min_link_prob: float | None = None,
        min_prior_prob: float | None = None,
        min_link_count: int | None = None,
    ) -> "DictionaryEntityLinker":
        data = np.load(os.path.join(data_dir, "data.npy"))
        offsets = np.load(os.path.join(data_dir, "offsets.npy"))
        name_trie = Trie()
        name_trie.load(os.path.join(data_dir, "name.trie"))
        kb_id_trie = Trie()
        kb_id_trie.load(os.path.join(data_dir, "kb_id.trie"))

        with open(os.path.join(data_dir, "config.json")) as config_file:
            config = json.load(config_file)

        if min_link_prob is None:
            min_link_prob = config["min_link_prob"]

        if min_prior_prob is None:
            min_prior_prob = config["min_prior_prob"]

        if min_link_count is None:
            min_link_count = config["min_link_count"]

        return DictionaryEntityLinker(
            name_trie=name_trie,
            kb_id_trie=kb_id_trie,
            data=data,
            offsets=offsets,
            max_mention_length=config["max_mention_length"],
            case_sensitive=config["case_sensitive"],
            min_link_prob=min_link_prob,
            min_prior_prob=min_prior_prob,
            min_link_count=min_link_count,
        )

    def detect_mentions(self, text: str) -> list[DictionaryELMention]:
        tokens = self._tokenizer(text)
        end_offsets = frozenset(token.idx + len(token) for token in tokens)
        if not self._case_sensitive:
            text = text.lower()

        ret = []
        cur = 0
        for token in tokens:
            start = token.idx
            if cur > start:
                continue

            for prefix in sorted(
                self._name_trie.prefixes(text[start : start + self._max_mention_length]), key=len, reverse=True
            ):
                end = start + len(prefix)
                if end in end_offsets:
                    matched = False
                    mention_idx = self._name_trie[prefix]
                    data_start, data_end = self._offsets[mention_idx : mention_idx + 2]
                    for kb_idx, link_count, total_link_count, doc_count in self._data[data_start:data_end]:
                        mention = DictionaryELMention(
                            kb_id=self._kb_id_trie.restore_key(kb_idx),
                            text=prefix,
                            start=start,
                            end=end,
                            link_count=link_count,
                            total_link_count=total_link_count,
                            doc_count=doc_count,
                        )
                        if (
                            mention.link_prob >= self._min_link_prob
                            and mention.prior_prob >= self._min_prior_prob
                            and mention.link_count >= self._min_link_count
                        ):
                            ret.append(mention)

                        matched = True

                    if matched:
                        cur = end
                        break

        return ret

    def detect_mentions_batch(self, texts: list[str]) -> list[list[DictionaryELMention]]:
        return [self.detect_mentions(text) for text in texts]

    def query(self, text: str) -> list[DictionaryELMention]:
        ret = []
        if not self._case_sensitive:
            text = text.lower()

        if text in self._name_trie:
            mention_idx = self._name_trie[text]
            data_start, data_end = self._offsets[mention_idx : mention_idx + 2]

            for kb_idx, link_count, total_link_count, doc_count in self._data[data_start:data_end]:
                mention = DictionaryELMention(
                    kb_id=self._kb_id_trie.restore_key(kb_idx),
                    text=text,
                    start=0,
                    end=len(text),
                    link_count=link_count,
                    total_link_count=total_link_count,
                    doc_count=doc_count,
                )
                ret.append(mention)

        return ret
