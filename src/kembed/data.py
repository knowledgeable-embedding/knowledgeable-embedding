import glob
import logging
import random
from dataclasses import dataclass
from typing import NamedTuple

import datasets
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

from kembed.arguments import DataArguments

logger = logging.getLogger(__name__)


class KPRTrainingItem(NamedTuple):
    query: BatchEncoding
    passages: list[BatchEncoding]


class KPRBaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _create_example(self, inputs: dict[str, list[int]], max_len: int) -> BatchEncoding:
        encoded_inputs = self.tokenizer.prepare_for_model(
            inputs["input_ids"],
            add_special_tokens=True,
            padding=False,
            truncation="only_first",
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        encoded_inputs["entity_ids"] = []
        encoded_inputs["entity_start_positions"] = []
        encoded_inputs["entity_lengths"] = []
        for entity_id, start_position, length in zip(
            inputs["entity_ids"], inputs["entity_start_positions"], inputs["entity_lengths"]
        ):
            start_position = start_position + 1  # +1 for [CLS] token
            if start_position + length > max_len - 1:  # -1 for [SEP] token
                continue

            encoded_inputs["entity_ids"].append(entity_id)
            encoded_inputs["entity_start_positions"].append(start_position)
            encoded_inputs["entity_lengths"].append(length)

        return encoded_inputs


class KPRTrainingDataset(KPRBaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: DataArguments):
        super().__init__(tokenizer)

        self.args = args

        training_datasets = []
        for training_dataset_dir in glob.glob(args.training_dataset_dir):
            training_datasets.append(load_from_disk(training_dataset_dir))
        self._training_dataset = datasets.concatenate_datasets(training_datasets)

        query_datasets = []
        for query_dataset_dir in glob.glob(args.query_dataset_dir):
            query_datasets.append(load_from_disk(query_dataset_dir))
        self._query_dataset = datasets.concatenate_datasets(query_datasets)

        passage_datasets = []
        for passage_dataset_dir in glob.glob(args.passage_dataset_dir):
            passage_datasets.append(load_from_disk(passage_dataset_dir))
        self._passage_dataset = datasets.concatenate_datasets(passage_datasets)

        self._query_id_mapping = {id_: i for i, id_ in enumerate(self._query_dataset["id"])}
        self._passage_id_mapping = {id_: i for i, id_ in enumerate(self._passage_dataset["id"])}

        self._query_prefix_token_ids = []
        if args.query_prefix:
            logger.info(f"query prefix: {args.query_prefix}")
            self._query_prefix_token_ids = tokenizer.encode(args.query_prefix, add_special_tokens=False)
        self._passage_prefix_token_ids = []
        if args.passage_prefix:
            logger.info(f"passage prefix: {args.passage_prefix}")
            self._passage_prefix_token_ids = tokenizer.encode(args.passage_prefix, add_special_tokens=False)

    def __len__(self):
        return len(self._training_dataset)

    def __getitem__(self, index: int) -> KPRTrainingItem:
        item = self._training_dataset[index]
        query_id = item["query_id"]
        query = self._query_dataset[self._query_id_mapping[query_id]]
        query["input_ids"] = self._query_prefix_token_ids + query["input_ids"]
        query["entity_start_positions"] = [
            pos + len(self._query_prefix_token_ids) for pos in query["entity_start_positions"]
        ]
        query_example = self._create_example(query, self.args.query_max_len)

        positive_passage_ids = item["positive_passage_ids"]
        positive_passage_id = random.choice(positive_passage_ids)
        positive_passage = self._passage_dataset[self._passage_id_mapping[positive_passage_id]]

        negative_passage_ids = item["negative_passage_ids"]
        negative_passages = [self._passage_dataset[self._passage_id_mapping[id_]] for id_ in negative_passage_ids]
        if self.args.sample_negatives_from_topk is not None:
            negative_passages = negative_passages[: self.args.sample_negatives_from_topk]

        if len(negative_passages) < self.args.train_group_size - 1:
            additional_negative_passage_ids = random.sample(
                self._passage_id_mapping.keys(), k=self.args.train_group_size - 1 - len(negative_passages)
            )
            negative_passages.extend(
                [self._passage_dataset[self._passage_id_mapping[id_]] for id_ in additional_negative_passage_ids]
            )
        else:
            negative_passages = random.sample(negative_passages, k=self.args.train_group_size - 1)

        passage_examples = []
        for passage in [positive_passage] + negative_passages:
            passage["input_ids"] = self._passage_prefix_token_ids + passage["input_ids"]
            passage["entity_start_positions"] = [
                pos + len(self._passage_prefix_token_ids) for pos in passage["entity_start_positions"]
            ]
            passage_examples.append(self._create_example(passage, self.args.passage_max_len))

        return KPRTrainingItem(query=query_example, passages=passage_examples)


class KPRInferenceDataset(KPRBaseDataset):
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, max_len=int, prefix: str = ""):
        super().__init__(tokenizer)

        self.dataset = dataset
        self._max_len = max_len
        self._prefix_token_ids = []
        if prefix:
            logger.info(f"prefix: {prefix}")
            self._prefix_token_ids = tokenizer.encode(prefix, add_special_tokens=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> BatchEncoding:
        item = self.dataset[index]
        item["input_ids"] = self._prefix_token_ids + item["input_ids"]
        item["entity_start_positions"] = [pos + len(self._prefix_token_ids) for pos in item["entity_start_positions"]]
        return self._create_example(item, max_len=self._max_len)


@dataclass
class KPRBaseCollator:
    tokenizer: PreTrainedTokenizer

    def _collate(self, inputs: list[BatchEncoding], max_len: int) -> BatchEncoding:
        entity_ids_list = [item.pop("entity_ids") for item in inputs]
        entity_start_positions_list = [item.pop("entity_start_positions") for item in inputs]
        entity_lengths_list = [item.pop("entity_lengths") for item in inputs]

        ret = self.tokenizer.pad(inputs, padding="max_length", max_length=max_len, return_tensors="pt")

        batch_size = len(entity_ids_list)
        max_entity_length_in_batch = max(len(ids) for ids in entity_ids_list)
        if max_entity_length_in_batch == 0:
            max_entity_length_in_batch = 1
            max_entity_token_length = 1
        else:
            max_entity_token_length = max(length for lengths in entity_lengths_list for length in lengths)
            max_entity_token_length = max(max_entity_token_length, 1)

        ret["entity_ids"] = torch.zeros(batch_size, max_entity_length_in_batch, dtype=torch.long)
        ret["entity_position_ids"] = torch.zeros(
            batch_size, max_entity_length_in_batch, max_entity_token_length, dtype=torch.long
        )

        for i, (entity_ids, entity_start_positions, entity_lengths) in enumerate(
            zip(entity_ids_list, entity_start_positions_list, entity_lengths_list)
        ):
            ret["entity_ids"][i, : len(entity_ids)] = torch.tensor(entity_ids)
            for j, (start_position, length) in enumerate(zip(entity_start_positions, entity_lengths)):
                ret["entity_position_ids"][i, j, :length] = torch.arange(start_position, start_position + length)

        return ret


@dataclass
class KPRTrainingCollator(KPRBaseCollator):
    query_max_len: int
    passage_max_len: int

    def __call__(self, items: list[KPRTrainingItem]) -> dict[str, BatchEncoding]:
        queries = [item.query for item in items]
        passages = [passage for item in items for passage in item.passages]

        collated_queries = self._collate(queries, max_len=self.query_max_len)
        collated_passages = self._collate(passages, max_len=self.passage_max_len)

        return {"queries": collated_queries, "passages": collated_passages}


@dataclass
class KPRInferenceCollator(KPRBaseCollator):
    max_len: int
    no_entities: bool = False

    def __call__(self, items: list[BatchEncoding]) -> dict[str, BatchEncoding]:
        collated_items = self._collate(items, max_len=self.max_len)
        if self.no_entities:
            collated_items.pop("entity_ids", None)
            collated_items.pop("entity_position_ids", None)

        return collated_items
