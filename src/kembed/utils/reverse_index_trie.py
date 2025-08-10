from collections.abc import Iterator
from typing import NamedTuple

from marisa_trie import RecordTrie


class ReverseIndexEntry(NamedTuple):
    index: int
    mention_offset: int


class ReverseIndexTrie:
    def __init__(self, trie: RecordTrie):
        self._trie = trie

    def __getitem__(self, key: str) -> list[ReverseIndexEntry]:
        entries = self._trie.get(key)
        if entries is None:
            return []

        ret = [ReverseIndexEntry(index, mention_offset) for index, mention_offset in entries]
        return ret

    def __contains__(self, key: str) -> bool:
        return key in self._trie

    def __len__(self) -> int:
        return len(self._trie)

    def keys(self) -> Iterator[str]:
        prev_kb_id = None
        for kb_id in self._trie.iterkeys():
            if kb_id == prev_kb_id:
                continue
            prev_kb_id = kb_id
            yield kb_id

    def items(self) -> Iterator[tuple[str, list[ReverseIndexEntry]]]:
        for kb_id in self.keys():
            yield kb_id, self[kb_id]

    @staticmethod
    def load(path: str) -> "ReverseIndexTrie":
        trie = RecordTrie("<II")
        trie.load(path)
        return ReverseIndexTrie(trie)

    def save(self, path: str) -> None:
        self._trie.save(path)

    @staticmethod
    def build(keys: list[str], values: list[ReverseIndexEntry]) -> "ReverseIndexTrie":
        trie = RecordTrie("<II", zip(keys, values))
        return ReverseIndexTrie(trie)
