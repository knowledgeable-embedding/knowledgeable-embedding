import numpy as np
import pytest
import torch

from kembed.embedder.bert import BertEmbedder

LAYER = 6
MAX_SEQ_LENGTH = 16


@pytest.fixture(scope="module")
def mask_embedder():
    return BertEmbedder("bert-base-uncased", strategy="mask", layer=LAYER, max_seq_length=MAX_SEQ_LENGTH, device="cpu")


@pytest.fixture(scope="module")
def average_embedder():
    return BertEmbedder(
        "bert-base-uncased", strategy="average", layer=LAYER, max_seq_length=MAX_SEQ_LENGTH, device="cpu"
    )


def _create_input_string(pre_ctx_len: int, post_ctx_len: int, target_len: int):
    pre_ctx_str = " ".join(str(n) for n in range(pre_ctx_len - 1, -1, -1))
    post_ctx_str = " ".join(str(n) for n in range(post_ctx_len))
    target_str = " ".join(["*"] * target_len)
    return f"{pre_ctx_str} {target_str} {post_ctx_str}", len(pre_ctx_str) + 1, len(pre_ctx_str) + 1 + len(target_str)


def test_embed(mask_embedder: BertEmbedder, average_embedder: BertEmbedder):
    input_str, char_start, char_end = _create_input_string(3, 3, 2)

    mask_input_str = input_str.replace("* *", "[MASK]")
    inputs = mask_embedder.tokenizer([mask_input_str], return_tensors="pt")
    with torch.inference_mode():
        outputs = mask_embedder.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[LAYER][0, 4].numpy()
        assert np.allclose(mask_embedder.embed(input_str, (char_start, char_end)), embedding)

    inputs = average_embedder.tokenizer([input_str], return_tensors="pt")
    with torch.inference_mode():
        outputs = average_embedder.model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[LAYER][0, 4:6].numpy().mean(axis=0)
        assert np.allclose(average_embedder.embed(input_str, (char_start, char_end)), embedding)


def test_preprocess(mask_embedder: BertEmbedder, average_embedder: BertEmbedder):
    # entire context string fits in the max_seq_length
    input_str, char_start, char_end = _create_input_string(5, 5, 2)
    tokens, start, end = mask_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "4", "3", "2", "1", "0", "[MASK]", "0", "1", "2", "3", "4", "[SEP]"]
    assert start == 6
    assert end == 7

    tokens, start, end = average_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "4", "3", "2", "1", "0", "*", "*", "0", "1", "2", "3", "4", "[SEP]"]
    assert start == 6
    assert end == 8

    # the context string before the target string is longer
    input_str, char_start, char_end = _create_input_string(10, 5, 2)
    tokens, start, end = mask_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "7", "6", "5", "4", "3", "2", "1", "0", "[MASK]", "0", "1", "2", "3", "4", "[SEP]"]
    assert start == 9
    assert end == 10

    input_str, char_start, char_end = _create_input_string(10, 5, 2)
    tokens, start, end = average_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "6", "5", "4", "3", "2", "1", "0", "*", "*", "0", "1", "2", "3", "4", "[SEP]"]
    assert start == 8
    assert end == 10

    # the context string after the target string is longef
    input_str, char_start, char_end = _create_input_string(5, 10, 2)
    tokens, start, end = mask_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "4", "3", "2", "1", "0", "[MASK]", "0", "1", "2", "3", "4", "5", "6", "7", "[SEP]"]
    assert start == 6
    assert end == 7

    input_str, char_start, char_end = _create_input_string(5, 10, 2)
    tokens, start, end = average_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "4", "3", "2", "1", "0", "*", "*", "0", "1", "2", "3", "4", "5", "6", "[SEP]"]
    assert start == 6
    assert end == 8

    # the both context strings are long
    input_str, char_start, char_end = _create_input_string(10, 10, 2)
    tokens, start, end = mask_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "6", "5", "4", "3", "2", "1", "0", "[MASK]", "0", "1", "2", "3", "4", "5", "[SEP]"]
    assert start == 8
    assert end == 9

    input_str, char_start, char_end = _create_input_string(10, 10, 2)
    tokens, start, end = average_embedder._preprocess(input_str, (char_start, char_end))
    assert tokens == ["[CLS]", "5", "4", "3", "2", "1", "0", "*", "*", "0", "1", "2", "3", "4", "5", "[SEP]"]
    assert start == 7
    assert end == 9


def test_create_inputs(mask_embedder: BertEmbedder, average_embedder: BertEmbedder):
    input_str, char_start, char_end = _create_input_string(5, 5, 2)
    inputs = mask_embedder.create_inputs([input_str], [(char_start, char_end)])
    assert mask_embedder.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "[CLS]",
        "4",
        "3",
        "2",
        "1",
        "0",
        "[MASK]",
        "0",
        "1",
        "2",
        "3",
        "4",
        "[SEP]",
    ]
    assert inputs["attention_mask"][0].tolist() == [1] * 13
    assert inputs["mention_mask"][0].tolist() == [False] * 6 + [True] * 1 + [False] * 6

    inputs = average_embedder.create_inputs([input_str], [(char_start, char_end)])
    assert mask_embedder.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "[CLS]",
        "4",
        "3",
        "2",
        "1",
        "0",
        "*",
        "*",
        "0",
        "1",
        "2",
        "3",
        "4",
        "[SEP]",
    ]
    assert inputs["attention_mask"][0].tolist() == [1] * 14
    assert inputs["mention_mask"][0].tolist() == [False] * 6 + [True] * 2 + [False] * 6
