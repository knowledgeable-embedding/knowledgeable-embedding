import argparse
import json
from collections import defaultdict


def main(args: argparse.Namespace):
    data = defaultdict(list)
    with open(args.results_file) as f:
        for line in f:
            item = json.loads(line)
            subset = item["subset"]
            data[subset].append(item["first_correct_passage_index"])
    for top_k in (20, 100):
        print(f"Top {top_k}:")
        accuracies = []
        for subset, indices in data.items():
            num_correct = sum(1 for index in indices if index is not None and index < top_k)
            accuracy = num_correct / len(indices) * 100
            accuracies.append(accuracy)
            print(f"{subset}\t{accuracy:.2f}%")
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average accuracy: {avg_accuracy:.2f}%")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSONL file")

    args = parser.parse_args()

    main(args)
