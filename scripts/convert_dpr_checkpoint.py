import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer


def main(args: argparse.Namespace):
    checkpoint_data = torch.load(args.checkpoint_file, map_location="cpu")
    config_name = checkpoint_data["encoder_params"]["encoder"]["pretrained_model_cfg"]
    model_dict = checkpoint_data["model_dict"]

    question_keys = [k for k in model_dict.keys() if k.startswith("question_model")]
    ctx_keys = [k for k in model_dict.keys() if k.startswith("ctx_model")]

    question_dict = dict([(k[len("question_model") + 1 :], model_dict[k]) for k in question_keys])
    ctx_dict = dict([(k[len("ctx_model") + 1 :], model_dict[k]) for k in ctx_keys])

    os.makedirs(os.path.join(args.output_dir, "query_model"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "passage_model"), exist_ok=True)

    config = AutoConfig.from_pretrained(config_name)
    tokenizer = AutoTokenizer.from_pretrained(config_name)

    query_model_dir = os.path.join(args.output_dir, "query_model")
    config.save_pretrained(query_model_dir)
    tokenizer.save_pretrained(query_model_dir)
    torch.save(question_dict, os.path.join(query_model_dir, "pytorch_model.bin"))

    passage_model_dir = os.path.join(args.output_dir, "passage_model")
    config.save_pretrained(passage_model_dir)
    tokenizer.save_pretrained(passage_model_dir)
    torch.save(ctx_dict, os.path.join(passage_model_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted model files")

    args = parser.parse_args()

    main(args)
