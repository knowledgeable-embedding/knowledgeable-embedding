import argparse
import glob
import json
import logging
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import datasets
import numpy as np
import torch
import transformers
from tqdm import tqdm

from kembed.retriever import Retriever
from kembed.utils.msmarco_eval import compute_metrics, load_reference
from kembed.utils.qa_eval import has_answer

logger = logging.getLogger(__name__)


def _get_first_passage_index_with_answer(passages: list[str], answers: list[str], regex: bool) -> int | None:
    for index, passage in enumerate(passages):
        for answer in answers:
            if has_answer(passage, answer, regex):
                return index
    return None


def main(args: argparse.Namespace):
    passage_embedding_files = glob.glob(args.passage_embedding_file)

    logger.info(f"Starting to load passage embeddings...")
    passage_embeddings_list = []
    for passage_embedding_file in tqdm(sorted(passage_embedding_files)):
        passage_embeddings = torch.load(passage_embedding_file, weights_only=True)
        passage_embeddings_list.append(passage_embeddings)
    passage_embeddings = torch.cat(passage_embeddings_list, dim=0)
    del passage_embeddings_list

    retriever = Retriever(normalize=args.normalize)
    retriever.build(passage_embeddings)
    total_num_passages = passage_embeddings.size(0)
    del passage_embeddings

    logger.info("Starting to load query embeddings...")
    query_embedding_files = glob.glob(args.query_embedding_file)
    query_embeddings_list = []
    for query_embedding_file in tqdm(sorted(query_embedding_files)):
        query_embeddings = torch.load(query_embedding_file, weights_only=True)
        query_embeddings_list.append(query_embeddings)
    query_embeddings = torch.cat(query_embeddings_list, dim=0)
    del query_embeddings_list

    logger.info("Starting to search...")
    search_result = retriever.search(query_embeddings, args.k)

    logger.info("Starting to load dataset...")
    passage_dataset = datasets.load_from_disk(args.passage_dataset_dir)
    passage_ids = passage_dataset["id"]
    assert len(passage_ids) == total_num_passages

    query_dataset = datasets.load_from_disk(args.query_dataset_dir)
    query_ids = query_dataset["id"]
    assert len(query_ids) == query_embeddings.shape[0]

    logger.info("Starting to compute metrics...")
    if args.mode == "msmarco":
        assert args.k >= 1000
        assert args.qrels_file is not None

        output_file = None
        if args.output_file is not None:
            output_file = open(args.output_file, "w")

        query_id_to_candidate_passage_ids = defaultdict(list)
        for index, query_id in enumerate(query_ids):
            scores = search_result.scores[index]
            indices = search_result.indices[index]
            # MEMO: maybe unnecessary
            sorted_indices_scores = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
            for passage_index, score in sorted_indices_scores:
                passage_id = passage_ids[passage_index]
                if output_file is not None:
                    output_file.write(f"{query_id}\t{passage_id}\t{score}\n")
                query_id_to_candidate_passage_ids[query_id].append(passage_id)

        if output_file is not None:
            output_file.close()

        qids_to_ranked_candidate_passages = {
            int(qid): [int(id_) for id_ in passage_ids[:1000]]
            for qid, passage_ids in query_id_to_candidate_passage_ids.items()
        }
        qids_to_relevant_passageids = load_reference(args.qrels_file)

        metrics = compute_metrics(
            qids_to_relevant_passageids=qids_to_relevant_passageids,
            qids_to_ranked_candidate_passages=qids_to_ranked_candidate_passages,
        )

    elif args.mode == "qa":
        output_file = None
        if args.output_file is not None:
            output_file = open(os.path.join(args.output_file), "w")

        answers_list = query_dataset["answers"]
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            first_correct_passage_indices = []
            for index, answers in enumerate(answers_list):
                scores = search_result.scores[index]
                indices = search_result.indices[index]
                sorted_indices = indices[np.argsort(-scores)]
                candidate_passages = [passage_dataset[int(i)]["text"] for i in sorted_indices]

                future = executor.submit(
                    _get_first_passage_index_with_answer, candidate_passages, answers, args.use_regex
                )
                first_correct_passage_indices.append(future)

            first_correct_passage_indices = [future.result() for future in first_correct_passage_indices]
            if output_file is not None:
                for query_item, first_correct_passage_index in zip(query_dataset, first_correct_passage_indices):
                    output_item = query_item.copy()
                    output_item["first_correct_passage_index"] = first_correct_passage_index
                    output_file.write(json.dumps(output_item) + "\n")

            metrics = []
            for top_k in (1, 5, 20, 50, 100):
                correct_count = 0
                for first_correct_passage_index in first_correct_passage_indices:
                    if first_correct_passage_index is not None and first_correct_passage_index < top_k:
                        correct_count += 1

                metrics.append((f"recall@{top_k}", correct_count / len(query_ids)))

        if output_file is not None:
            output_file.close()

    logging.info(f"Metrics: {metrics}")

    if args.wandb_run_id is not None:
        import wandb
        import wandb.errors

        for _ in range(10):
            try:
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    id=args.wandb_run_id,
                    name=args.wandb_run_name,
                    resume="allow",
                )
                break
            except wandb.errors.UsageError:
                time.sleep(60)
                continue

        log_metrics = {f"{key}{args.wandb_metric_suffix}": value for key, value in metrics}
        run.log(log_metrics)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_embedding_file", type=str, required=True, help="Path or glob pattern to query embedding file(s)"
    )
    parser.add_argument(
        "--passage_embedding_file", type=str, required=True, help="Path or glob pattern to passage embedding file(s)"
    )
    parser.add_argument("--query_dataset_dir", type=str, required=True, help="Path to query dataset directory")
    parser.add_argument("--passage_dataset_dir", type=str, required=True, help="Path to passage dataset directory")
    parser.add_argument("--output_file", type=str, help="Path to output file for search results")
    parser.add_argument("--qrels_file", type=str, help="Path to qrels TSV file (required for msmarco mode)")
    parser.add_argument("--k", type=int, default=1000, help="Number of top passages to retrieve")
    parser.add_argument("--normalize", action="store_true", help="Normalize embeddings before search")
    parser.add_argument("--use_regex", action="store_true", help="Use regex matching for answers in QA mode")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for retrieval")
    parser.add_argument(
        "--max_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument("--mode", type=str, default="msmarco", choices=["msmarco", "qa"], help="Evaluation mode")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, help="Weights & Biases entity name")
    parser.add_argument("--wandb_run_id", type=str, help="Weights & Biases run ID to log to")
    parser.add_argument("--wandb_run_name", type=str, help="Weights & Biases run name")
    parser.add_argument("--wandb_metric_suffix", type=str, default="", help="Suffix to append to logged metric names")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    transformers.utils.logging.set_verbosity(logging.WARNING)

    main(args)
