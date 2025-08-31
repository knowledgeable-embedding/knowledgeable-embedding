import argparse
import contextlib
import logging
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from transformers import AutoModel, AutoTokenizer

import kembed
from kembed.utils import load_tsv_mapping

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    temp_dir_cm = tempfile.TemporaryDirectory() if args.temp_dir is None else contextlib.nullcontext(args.temp_dir)
    with temp_dir_cm as temp_dir:
        if model.config.entity_vocab_size is None:  # The model has already been converted with this script
            logger.info("The model has already been converted with this script.")
            entity_embedding_norm = None
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        else:  # The model has not been converted with this script
            logger.info("The model has not been converted with this script.")
            model.config.entity_vocab_size = None
            entity_embedding_norm = (
                model.entity_fusion_layer.entity_embeddings.embeddings.weight.data[-1].norm(p=2).item()
            )
            logger.info(f"Computed entity embedding norm: {entity_embedding_norm}")
            del model.entity_fusion_layer.entity_embeddings.embeddings

            model.config.register_for_auto_class()
            model.register_for_auto_class("AutoModel")

            tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_name_or_path, trust_remote_code=True)

        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        del model, tokenizer

        if os.path.exists(os.path.join(temp_dir, "entity_linker")):
            shutil.rmtree(os.path.join(temp_dir, "entity_linker"))
        shutil.copytree(args.entity_linker_dir, os.path.join(temp_dir, "entity_linker"))
        shutil.copy(args.entity_vocab_file, os.path.join(temp_dir, "entity_vocab.tsv"))

        if entity_embedding_norm is None:
            original_entity_embeddings = np.load(os.path.join(temp_dir, "entity_embeddings.npy"))
            entity_embedding_norm = np.linalg.norm(original_entity_embeddings[1], ord=2)
            logger.info(f"Computed entity embedding norm: {entity_embedding_norm}")

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

        np.save(os.path.join(temp_dir, "entity_embeddings.npy"), target_entity_embeddings.numpy())

        del entity_vocab, embedding_entity_vocab, entity_embeddings, target_entity_embeddings

        transformer = Transformer(
            temp_dir,
            model_args={"trust_remote_code": True},
            config_args={"trust_remote_code": True},
            tokenizer_args={"trust_remote_code": True},
        )
        pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")

        sentence_transformers_model = SentenceTransformer(
            modules=[transformer, pooling], similarity_fn_name=model.config.similarity_function, trust_remote_code=True
        )
        if args.output_dir:
            sentence_transformers_model.save_pretrained(args.output_dir)

    if args.hf_repo_id:
        sentence_transformers_model.push_to_hub(args.hf_repo_id, private=args.private, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path or name of the trained model to load"
    )
    parser.add_argument(
        "--base_tokenizer_name_or_path",
        type=str,
        default="knowledgeable-ai/kpr-bert-tokenizer",
        help="Hugging Face repository ID of the base tokenizer. Used only if the model has not yet been converted with this script.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory to save the model")
    parser.add_argument("--temp_dir", type=str, help="Temporary directory for intermediate files")
    parser.add_argument(
        "--entity_linker_dir", type=str, required=True, help="Path to the directory containing the entity linker data"
    )
    parser.add_argument("--entity_vocab_file", type=str, required=True, help="Path to the entity vocabulary TSV file")
    parser.add_argument(
        "--entity_embedding_dir", type=str, required=True, help="Path to the directory containing entity embeddings"
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
