import argparse
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from kembed.utils import load_tsv_mapping

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    path = snapshot_download(repo_id=args.base_tokenizer_repo_id)
    shutil.copytree(path, args.output_dir, dirs_exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "entity_linker")):
        shutil.rmtree(os.path.join(args.output_dir, "entity_linker"))
    shutil.copytree(args.entity_linker_dir, os.path.join(args.output_dir, "entity_linker"))
    shutil.copy(args.entity_vocab_file, os.path.join(args.output_dir, "entity_vocab.tsv"))

    if args.entity_embedding_dir is not None:
        if args.entity_embedding_norm is None:
            original_entity_embeddings = np.load(os.path.join(args.output_dir, "entity_embeddings.npy"))
            entity_embedding_norm = np.linalg.norm(original_entity_embeddings[1], ord=2)
            logger.info(f"Computed entity embedding norm: {entity_embedding_norm}")
        else:
            entity_embedding_norm = args.entity_embedding_norm

        entity_vocab = load_tsv_mapping(args.entity_vocab_file, int)

        entity_embeddings = torch.load(os.path.join(args.entity_embedding_dir, "embeddings.pt"), weights_only=True)
        entity_embedding_size = entity_embeddings.size(1)

        embedding_entity_vocab = load_tsv_mapping(os.path.join(args.entity_embedding_dir, "entity_vocab.tsv"), int)
        target_entity_embeddings = torch.zeros(len(entity_vocab), entity_embedding_size)
        for kb_id, index in entity_vocab.items():
            if index == 0:
                assert kb_id == "[PAD]"
                continue
            target_entity_embeddings[index] = entity_embeddings[embedding_entity_vocab[kb_id]]

        target_entity_embeddings = F.normalize(target_entity_embeddings, p=2, dim=-1)
        target_entity_embeddings = target_entity_embeddings * entity_embedding_norm
        target_entity_embeddings = target_entity_embeddings.half()

        np.save(os.path.join(args.output_dir, "entity_embeddings.npy"), target_entity_embeddings.numpy())

    if args.hf_repo_id:
        new_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
        new_tokenizer.push_to_hub(args.hf_repo_id, private=args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_tokenizer_repo_id",
        type=str,
        default="studio-ousia/kpr-bert-tokenizer",
        help="Hugging face repository ID of the tokenizer to use as the base",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the tokenizer")
    parser.add_argument(
        "--entity_linker_dir", type=str, required=True, help="Path to the directory containing the entity linker data"
    )
    parser.add_argument("--entity_vocab_file", type=str, required=True, help="Path to the entity vocabulary TSV file")
    parser.add_argument(
        "--entity_embedding_dir", type=str, help="Path to the directory containing entity embeddings (optional)"
    )
    parser.add_argument(
        "--entity_embedding_norm",
        type=float,
        help="L2 norm value to scale entity embeddings. If not provided, it will be computed from the base tokenizer's entity embeddings.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="Repository ID on the Hugging Face Hub to push the model to (e.g., username/repo_name)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Set the uploaded repository as private on the Hugging Face Hub"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
