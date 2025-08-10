from abc import ABCMeta, abstractmethod

import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding


class BaseEmbedder(metaclass=ABCMeta):
    def __init__(self, model_name_or_path: str, layer: int, max_seq_length: int, device: str = "cuda"):
        self._model_name_or_path = model_name_or_path
        self._layer = layer
        self._max_seq_length = max_seq_length
        self._device = device

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self._model_name_or_path)
            self._model.eval()
            self._model.to(self._device)
        return self._model

    def embed(self, text: str, span: tuple[int, int]) -> torch.Tensor:
        return self.embed_batch([text], [span])[0]

    def embed_batch(self, texts: list[str], spans: list[tuple[int, int]]) -> torch.Tensor:
        inputs = self.create_inputs(texts, spans)
        embeddings = self.compute_embeddings(inputs)

        return embeddings

    @abstractmethod
    def create_inputs(self, texts: list[str], spans: list[tuple[int, int]]) -> BatchEncoding:
        pass

    def compute_embeddings(self, inputs: BatchEncoding) -> torch.Tensor:
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            mention_mask = inputs.pop("mention_mask")
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[self._layer]
            mention_mask = mention_mask.type_as(hidden_state).unsqueeze(-1)
            hidden_state = hidden_state * mention_mask
            embeddings = hidden_state.sum(dim=1) / mention_mask.sum(dim=1)

        return embeddings
