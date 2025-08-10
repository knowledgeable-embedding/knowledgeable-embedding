import argparse
import logging
import multiprocessing
import os
import random
from collections import deque
from datetime import timedelta

import datasets
import torch
import transformers
from accelerate import PartialState
from tqdm import trange
from tqdm.contrib import tenumerate
from transformers import BatchEncoding

from kembed.embedder.base import BaseEmbedder
from kembed.embedder.bert import BertEmbedder
from kembed.utils import ReverseIndexEntry, ReverseIndexTrie, load_tsv_mapping

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()

_dataset: datasets.Dataset
_embedder: BaseEmbedder


def _init_worker(dataset: datasets.Dataset, embedder: BaseEmbedder) -> None:
    global _dataset, _embedder
    _dataset = dataset
    _embedder = embedder


def _create_inputs(kb_ids: list[str], entries: list[ReverseIndexEntry]) -> tuple[list[str], BatchEncoding]:
    texts, spans = [], []
    for entry in entries:
        text = _dataset[entry.index]["text"]
        mention = _dataset[entry.index]["mentions"][entry.mention_offset]
        texts.append(text)
        spans.append((mention["start"], mention["end"]))

    return kb_ids, _embedder.create_inputs(texts, spans)


def main(args: argparse.Namespace) -> None:
    distributed_state = PartialState(timeout=timedelta(hours=3))  # long timeout for the final synchronization
    random.seed(args.seed)

    dataset = datasets.load_from_disk(args.preprocessed_dataset_dir)
    dataset.select_columns(["text", "mentions"])

    embedder = BertEmbedder(
        model_name_or_path=args.model_name_or_path,
        strategy=args.strategy,
        layer=args.layer,
        max_seq_length=args.max_seq_length,
        device=distributed_state.device,
    )
    entity_reverse_index = ReverseIndexTrie.load(args.entity_reverse_index_file)

    entity_vocab = {
        kb_id: n for n, (kb_id, items) in enumerate(entity_reverse_index.items()) if len(items) >= args.min_count
    }

    if distributed_state.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    batch_size = args.batch_size
    all_embeddings: torch.Tensor = None

    def _update_entity_embeddings(kb_ids: list[str], inputs: BatchEncoding) -> None:
        embeddings = embedder.compute_embeddings(inputs).to("cpu")

        nonlocal all_embeddings
        if all_embeddings is None:
            all_embeddings = torch.zeros(len(entity_vocab), embeddings.shape[1], dtype=torch.float32)

        for n, kb_id in enumerate(kb_ids):
            all_embeddings[entity_vocab[kb_id]] += embeddings[n]

    pool = multiprocessing.Pool(args.max_workers, initializer=_init_worker, initargs=(dataset, embedder))
    input_queue = deque()
    kb_id_buf, entry_buf = [], []
    batch_index = 0
    for kb_id_index, kb_id in tenumerate(entity_vocab.keys(), disable=not distributed_state.is_main_process):
        entries = entity_reverse_index[kb_id]
        if len(entries) > args.max_examples:
            entries = random.sample(entries, args.max_examples)

        for entry_index, entry in enumerate(entries):
            kb_id_buf.append(kb_id)
            entry_buf.append(entry)
            if len(entry_buf) == batch_size or (
                kb_id_index == len(entity_vocab) - 1 and entry_index == len(entries) - 1
            ):
                if batch_index % distributed_state.num_processes == distributed_state.process_index:
                    input_queue.append(pool.apply_async(_create_inputs, (kb_id_buf, entry_buf)))
                kb_id_buf, entry_buf = [], []
                batch_index += 1

        while len(input_queue) >= 10 or (input_queue and kb_id_index == len(entity_vocab) - 1):
            kb_ids, inputs = input_queue.popleft().get()
            _update_entity_embeddings(kb_ids, inputs)

    torch.save(all_embeddings, os.path.join(args.output_dir, f"embeddings_{distributed_state.process_index:02d}.pt"))
    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        all_embeddings = torch.zeros(len(entity_vocab), all_embeddings.shape[1], dtype=torch.float32)
        for process_index in trange(distributed_state.num_processes):
            embedding_file = os.path.join(args.output_dir, f"embeddings_{process_index:02d}.pt")
            embeddings = torch.load(embedding_file, map_location="cpu")
            all_embeddings += embeddings
            os.remove(embedding_file)

        torch.save(all_embeddings, os.path.join(args.output_dir, "embeddings.pt"))

        with open(os.path.join(args.output_dir, "entity_vocab.tsv"), "w") as f:
            for kb_id, index in entity_vocab.items():
                f.write(f"{kb_id}\t{index}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dataset_dir", type=str, required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument(
        "--entity_reverse_index_file", type=str, required=True, help="Path to entity reverse index file"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
    parser.add_argument(
        "--strategy", type=str, choices=["mask", "average"], default="mask", help="Embedding extraction strategy"
    )
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to extract embeddings from")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_examples", type=int, default=128, help="Maximum examples per entity")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum occurrence count to include entity")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    multiprocessing.set_start_method("spawn")

    main(args)
