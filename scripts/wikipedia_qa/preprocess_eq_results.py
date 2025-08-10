import argparse
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    with open(args.output_file, "w") as output_file:
        for json_file in glob.glob(os.path.join(args.eq_output_dir, "*.json")):
            logger.info(f"Loading {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
                subset = os.path.basename(json_file).split(".")[0]
                for example_id, item in enumerate(data):
                    first_correct_passage_index = None
                    for ctx_id, ctx in enumerate(item["ctxs"]):
                        if ctx["has_answer"]:
                            first_correct_passage_index = ctx_id
                            break

                    json_item = {
                        "id": f"{subset}_{example_id}",
                        "text": item["question"],
                        "answers": item["answers"],
                        "subset": subset,
                        "first_correct_passage_index": first_correct_passage_index,
                    }
                    output_file.write(json.dumps(json_item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eq_output_dir", type=str, required=True, help="Path to directory containing JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )

    main(args)
