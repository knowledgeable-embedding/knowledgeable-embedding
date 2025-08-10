import functools
import re
import unicodedata

import regex

# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L154
ALPHA_NUM = "[\p{L}\p{N}\p{M}]+"
NON_WS = "[^\p{Z}\p{C}]"
# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L163
REGEXP = regex.compile("(%s)|(%s)" % (ALPHA_NUM, NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)


def has_answer(passage: str, answer: str, regex=False) -> bool:
    if regex:
        passage = unicodedata.normalize("NFD", passage)
        answer = unicodedata.normalize("NFD", answer)
        answer_pattern = _compile_regex(answer)
        if answer_pattern is None:
            return False

        return answer_pattern.search(passage) is not None

    else:
        passage_tokens = _preprocess_and_tokenize(passage)
        answer_tokens = _preprocess_and_tokenize(answer)

        for i in range(0, len(passage_tokens) - len(answer_tokens) + 1):
            if answer_tokens == passage_tokens[i : i + len(answer_tokens)]:
                return True

        return False


@functools.lru_cache(maxsize=1000)
def _compile_regex(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return None


@functools.lru_cache(maxsize=1000)
def _preprocess_and_tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    return [m.group() for m in REGEXP.finditer(text)]
