import torch
from transformers import BatchEncoding

from .base import BaseEmbedder


class BertEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        strategy: str,
        layer: int,
        max_seq_length: int,
        device: str = "cuda",
    ):
        super().__init__(model_name_or_path, layer, max_seq_length, device)
        self._strategy = strategy
        assert self._strategy in ["mask", "average"], "strategy must be either mask or average"

    def create_inputs(self, texts: list[str], spans: list[tuple[int, int]]) -> BatchEncoding:
        input_ids = []
        mention_mask = torch.zeros(len(texts), self._max_seq_length, dtype=torch.bool)
        for n, (text, span) in enumerate(zip(texts, spans)):
            tokens, token_start, token_end = self._preprocess(text, span)
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            mention_mask[n, token_start:token_end] = True

        inputs = self.tokenizer.prepare_for_model(
            input_ids,
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        mention_mask = mention_mask[:, : inputs["input_ids"].shape[1]]
        inputs["mention_mask"] = mention_mask
        return inputs

    def _preprocess(self, text: str, span: tuple[int, int]) -> tuple[list[str], int, int]:
        start, end = span

        pre_tokens = []
        if start > 0:
            pre_text = text[:start].rstrip()
            pre_tokens = self.tokenizer.tokenize(pre_text)

        if self._strategy == "mask":
            mention_tokens = [self.tokenizer.mask_token]
        else:
            if start > 1 and text[start - 1] == " ":
                mention_text = text[start - 1 : end]  # for RoBERTa-like models
            else:
                mention_text = text[start:end]
            mention_tokens = self.tokenizer.tokenize(mention_text)

            if len(mention_tokens) == 0:
                # FIXME: this is a temporal workaround for the mention being empty
                mention_tokens = [self.tokenizer.mask_token]
            elif len(mention_text) > self._max_seq_length - 2:
                mention_tokens = mention_tokens[: self._max_seq_length - 2]

        max_context_length = self._max_seq_length - 2 - len(mention_tokens)
        half_context_length = max_context_length // 2

        post_tokens = []
        if end < len(text):
            post_text = text[end:]
            post_tokens = self.tokenizer.tokenize(post_text)

        if max_context_length == 0:
            pre_tokens = []
            post_tokens = []
        elif len(pre_tokens) < half_context_length:
            post_tokens = post_tokens[: max_context_length - len(pre_tokens)]
        elif len(post_tokens) < half_context_length:
            pre_tokens = pre_tokens[-(max_context_length - len(post_tokens)) :]
        else:
            post_tokens = post_tokens[:half_context_length]
            pre_tokens = pre_tokens[-(max_context_length - len(post_tokens)) :]

        tokens = [self.tokenizer.cls_token] + pre_tokens + mention_tokens + post_tokens + [self.tokenizer.sep_token]
        token_start = len(pre_tokens) + 1  # one for [CLS]
        token_end = token_start + len(mention_tokens)

        return tokens, token_start, token_end
