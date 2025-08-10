import argparse


def main(args: argparse.Namespace) -> None:
    with open(args.input_file) as f_in, open(args.output_file, "w") as f_out:
        cur_qid = None
        rank = 0
        for line in f_in:
            qid, docid, score = line.split()
            if cur_qid != qid:
                cur_qid = qid
                rank = 0
            rank += 1
            f_out.write(f"{qid}\t{docid}\t{rank}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")

    args = parser.parse_args()

    main(args)
