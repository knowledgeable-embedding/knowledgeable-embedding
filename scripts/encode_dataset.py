import argparse
import logging
import math
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

from kembed.data import KPRInferenceCollator, KPRInferenceDataset

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    distributed_state = PartialState()

    if args.tokenizer_name_or_path is None:
        tokenizer_name_or_path = args.model_name_or_path
    else:
        tokenizer_name_or_path = args.tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=False, trust_remote_code=True)

    if args.precision == 16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if args.model_type == "dpr":
        # We need to directly use model classes here because AutoModel wrongly detects context encoders as question encoders
        if "question" in args.model_name_or_path:
            model = DPRQuestionEncoder.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
        elif "ctx" in args.model_name_or_path:
            model = DPRContextEncoder.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
        else:
            raise ValueError(f"Invalid model_name_or_path: {args.model_name_or_path}")
    elif args.model_type in ("auto", "kpr"):
        model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    model = model.to(distributed_state.device)
    model.eval()

    if distributed_state.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_from_disk(args.dataset_dir)
    dataset_size_per_process = math.ceil(len(dataset) / distributed_state.num_processes)
    start_index = distributed_state.process_index * dataset_size_per_process
    end_index = min(start_index + dataset_size_per_process, len(dataset))

    if start_index >= end_index:
        distributed_state.wait_for_everyone()
        return

    dataset = dataset.select(range(start_index, end_index))

    all_embeddings = []
    inference_dataset = KPRInferenceDataset(
        tokenizer=tokenizer, dataset=dataset, max_len=args.max_len, prefix=args.prefix
    )

    no_entities = args.no_entities
    if args.model_type != "kpr":
        no_entities = True
    collator = KPRInferenceCollator(tokenizer=tokenizer, max_len=args.max_len, no_entities=no_entities)
    dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.max_dataloader_workers,
        pin_memory=True,
    )
    for inputs in tqdm(dataloader, disable=not distributed_state.is_main_process):
        inputs = {k: v.to(distributed_state.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if args.model_type == "dpr":
                embeddings = model(**inputs).pooler_output.cpu()
            elif args.model_type == "kpr":
                embeddings = model.encode(**inputs).cpu()
            elif args.model_type == "auto":
                model_outputs = model(**inputs)
                if args.pooling == "cls":
                    embeddings = model_outputs.last_hidden_state[:, 0].cpu()
                elif args.pooling == "mean":
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    last_hidden = model_outputs.last_hidden_state.masked_fill(~attention_mask.bool(), 0.0)
                    embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
                    embeddings = embeddings.cpu()
                else:
                    raise ValueError(f"Invalid pooling: {args.pooling}")
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    distributed_state.wait_for_everyone()

    torch.save(all_embeddings, os.path.join(args.output_dir, f"embeddings_{distributed_state.process_index:02d}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32], help="Precision for inference")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path or name of the tokenizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_dataloader_workers", type=int, default=8, help="Maximum number of dataloader workers")
    parser.add_argument(
        "--model_type", type=str, default="kpr", choices=["dpr", "auto", "kpr"], help="Type of model to use"
    )
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"], help="Pooling method")
    parser.add_argument("--no_entities", action="store_true", help="Disable entity input features in KPR models")
    parser.add_argument("--prefix", type=str, default="", help="Prefix to prepend to each input text")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    transformers.utils.logging.set_verbosity(logging.WARNING)

    main(args)
