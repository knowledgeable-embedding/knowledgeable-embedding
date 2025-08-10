"""
The original file is obtained from the following URL:
https://github.com/staoxiao/RetroMAE/blob/ba53e9efdaff6218ae4e93d9eb02f4f616813bcf/examples/retriever/msmarco/msmarco_eval.py

This module computes evaluation metrics for MSMARCO dataset on the ranking task. Intenral hard coded eval files version. DO NOT PUBLISH!
Command line:
python msmarco_eval_ranking.py <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 4/09/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""

import argparse
import json
from collections import Counter, defaultdict


def load_reference(path_to_reference: str) -> dict[int, list[int]]:
    """Load Reference reference relevant passages."""
    with open(path_to_reference) as f:
        qids_to_relevant_passageids = defaultdict(list)
        for l in f:
            qid, pid = l.strip().split("\t")
            qids_to_relevant_passageids[int(qid)].append(int(pid))

    return qids_to_relevant_passageids


def load_candidate(path_to_candidate: str) -> dict[int, list[int]]:
    """Load candidate data from a file."""
    with open(path_to_candidate) as f:
        # By default, all PIDs in the list of 1000 are 0. Only override those that are given
        qid_to_ranked_candidate_passages = defaultdict(lambda: [0] * 1000)
        for l in f:
            qid, pid, rank = l.strip().split("\t")
            qid_to_ranked_candidate_passages[int(qid)][int(rank) - 1] = int(pid)

    return qid_to_ranked_candidate_passages


def compute_metrics(
    qids_to_relevant_passageids: dict[int, list[int]], qids_to_ranked_candidate_passages: dict[int, list[int]]
) -> list[tuple[str, float | int]]:
    all_scores = []
    for max_mrr_rank in [10, 50, 100, 500, 1000]:
        mrr = 0
        acc = 0
        recall = 0
        ranking = []
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                for i in range(0, max_mrr_rank):
                    if candidate_pid[i] in target_pid:
                        acc += 1
                        mrr += 1 / (i + 1)
                        ranking.pop()
                        ranking.append(i + 1)
                        break
                recall += len(set(target_pid) & set(candidate_pid[:max_mrr_rank])) / len(target_pid)
        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

        mrr = mrr / len(qids_to_relevant_passageids)
        acc = acc / len(qids_to_relevant_passageids)
        recall = recall / len(qids_to_relevant_passageids)
        # all_scores.append(('ACC @{}'.format(MaxMRRRank), ACC))
        all_scores.append(("mrr@{}".format(max_mrr_rank), mrr))
        all_scores.append(("recall@{}".format(max_mrr_rank), recall))

    return all_scores


def compute_metrics_from_files(path_to_reference: str, path_to_candidate: str) -> list[tuple[str, float | int]]:
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1]
        )
        assert (
            len(duplicate_pids - set([0])) == 0
        ), f"Cannot rank a passage multiple times for a single query. QID={qid}, PID={list(duplicate_pids)[0]}"

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_reference", type=str, required=True)
    parser.add_argument("--path_to_candidate", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    metrics = compute_metrics_from_files(
        path_to_candidate=args.path_to_candidate, path_to_reference=args.path_to_reference
    )
    for x, y in metrics:
        print("{}: {}".format(x, y))

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(dict(metrics), f, indent=2)
