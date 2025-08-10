import re
from typing import NamedTuple

from transformers import PreTrainedTokenizerBase

from kembed.entity_linker.mention import BaseMention

WHITESPACE_RE = re.compile(r"\s+")


class _Entity(NamedTuple):
    entity_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def preprocess_text(
    text: str,
    mentions: list[BaseMention] | None,
    title: str | None,
    title_mentions: list[BaseMention] | None,
    tokenizer: PreTrainedTokenizerBase,
    entity_vocab: dict[str, int],
) -> dict[str, list[int]]:
    tokens = []
    entity_ids = []
    entity_start_positions = []
    entity_lengths = []
    if title is not None:
        if title_mentions is None:
            title_mentions = []

        title_tokens, title_entities = _tokenize_text_with_mentions(title, title_mentions, tokenizer, entity_vocab)
        tokens += title_tokens + [tokenizer.sep_token]
        for entity in title_entities:
            entity_ids.append(entity.entity_id)
            entity_start_positions.append(entity.start)
            entity_lengths.append(entity.end - entity.start)

    if mentions is None:
        mentions = []

    entity_offset = len(tokens)
    text_tokens, text_entities = _tokenize_text_with_mentions(text, mentions, tokenizer, entity_vocab)
    tokens += text_tokens
    for entity in text_entities:
        entity_ids.append(entity.entity_id)
        entity_start_positions.append(entity.start + entity_offset)
        entity_lengths.append(entity.end - entity.start)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "input_ids": input_ids,
        "entity_ids": entity_ids,
        "entity_start_positions": entity_start_positions,
        "entity_lengths": entity_lengths,
    }


def _tokenize_text_with_mentions(
    text: str, mentions: list[BaseMention], tokenizer: PreTrainedTokenizerBase, entity_vocab: dict[str, int]
) -> tuple[list[str], list[_Entity]]:
    target_mentions = [mention for mention in mentions if mention.kb_id is not None and mention.kb_id in entity_vocab]
    split_char_positions = {mention.start for mention in target_mentions} | {mention.end for mention in target_mentions}

    tokens = []
    cur = 0
    char_to_token_mapping = {}
    for char_position in sorted(split_char_positions):
        target_text = text[cur:char_position]
        tokens += tokenizer.tokenize(target_text)
        char_to_token_mapping[char_position] = len(tokens)
        cur = char_position
    tokens += tokenizer.tokenize(text[cur:])

    entities = [
        _Entity(entity_vocab[mention.kb_id], char_to_token_mapping[mention.start], char_to_token_mapping[mention.end])
        for mention in target_mentions
    ]
    return tokens, entities
