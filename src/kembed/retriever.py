from typing import NamedTuple

import numpy as np
import torch


class RetrieverResult(NamedTuple):
    scores: np.ndarray
    indices: np.ndarray


class Retriever:
    def __init__(self, devices: list[torch.device | str] | None = None, batch_size: int = 128, normalize: bool = False):
        if devices is not None:
            self._devices = devices
        else:
            self._devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self._batch_size = batch_size
        self._normalize = normalize

        self._passage_embeddings_list: list[torch.Tensor] = []

    def build(self, passage_embeddings: torch.Tensor) -> None:
        split_size = passage_embeddings.size(0) // len(self._devices)
        if self._normalize:
            passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=-1)
        for n, device in enumerate(self._devices):
            start = split_size * n
            self._passage_embeddings_list.append(passage_embeddings[start : start + split_size].to(device))

    def search(self, query_embeddings: torch.Tensor, k: int) -> RetrieverResult:
        all_scores = []
        all_indices = []
        for query_embeddings_batch in torch.split(query_embeddings, self._batch_size):
            scores_list = []
            for device, passage_embeddings in zip(self._devices, self._passage_embeddings_list):
                scores_list.append(torch.matmul(query_embeddings_batch.to(device), passage_embeddings.T))

            scores_list = [scores.to("cpu") for scores in scores_list]
            scores = torch.cat(scores_list, dim=1)
            scores, indices = torch.topk(scores, k, dim=1)
            all_scores.append(scores)
            all_indices.append(indices)

        scores = torch.cat(all_scores, dim=0)
        indices = torch.cat(all_indices, dim=0)

        return RetrieverResult(scores.numpy(), indices.numpy())
