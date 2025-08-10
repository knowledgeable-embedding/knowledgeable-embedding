import argparse

from transformers import AutoModel

import kembed


def main(args: argparse.Namespace) -> None:
    model = AutoModel.from_pretrained(args.model_name_or_path)
    if not args.with_entity_embeddings:
        model.config.entity_vocab_size = None
        del model.entity_fusion_layer.entity_embeddings.embeddings
    model.config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    model.push_to_hub(args.hf_repo_id, private=args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path or name of the trained model to load")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="Repository ID on the Hugging Face Hub to push the model to (e.g., username/repo_name)",
    )
    parser.add_argument(
        "--with_entity_embeddings", action="store_true", help="Include entity embeddings in the uploaded model"
    )
    parser.add_argument(
        "--private", action="store_true", help="Set the uploaded repository as private on the Hugging Face Hub"
    )
    args = parser.parse_args()

    main(args)
