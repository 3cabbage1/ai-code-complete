import json
import random
from typing import List

from pydantic import BaseModel
from loguru import logger
from coderag.config import settings
from coderag.retrieve.common import RetrieveResult
from coderag.benchmark.methods import load_benchmark
from operator import itemgetter


class ResultBlock(BaseModel):
    block_idx: int
    code_snippet: str


class RerankResultSaving(BaseModel):
    data_list: list[list[ResultBlock]]


def retrieve_block_filter(item: str) -> bool:
    return bool(item and item.strip())


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]


def build_norm_score_map(docs: list[str]) -> dict[str, float]:
    if not docs:
        return {}
    # Use rank position as raw score since retrieval output has no explicit score.
    raw_scores = [float(len(docs) - idx) for idx in range(len(docs))]
    norm_scores = normalize(raw_scores)
    return {doc: score for doc, score in zip(docs, norm_scores, strict=True)}


def normalize_doc_key(doc: str) -> str:
    return doc.strip()


def main():
    benchmark = load_benchmark()
    with open(settings.rerank.use_retrieve_file, "r", encoding="utf-8") as f:
        retrieve_result = RetrieveResult.model_validate_json(f.read())
    
    if settings.rerank.retrieve_data_indices_path is not None:
        with open(settings.rerank.retrieve_data_indices_path, "r", encoding="utf-8") as f:
            indices = json.load(f)
        retrieve_result.data_list = itemgetter(*indices)(retrieve_result.data_list)

    sample_n = settings.sample_n
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]
        retrieve_result.data_list = retrieve_result.data_list[:sample_n]


    result_blocks: List[List[ResultBlock]] = []
    alpha = settings.rerank.alpha
    top_k = settings.rerank.top_k
    use_sorted = settings.rerank.sorted

    for item in retrieve_result.data_list:
        sparse_items, dataflow_item = item.sparse or [], item.dataflow or []
        if not settings.rerank.use_sparse:
            sparse_items = []
        if not settings.rerank.use_dataflow:
            dataflow_item = []
        sparse_items = [it for it in sparse_items if retrieve_block_filter(it)]
        dataflow_item = [it for it in dataflow_item if retrieve_block_filter(it)]

        sparse_score_map = build_norm_score_map(sparse_items)
        dataflow_score_map = build_norm_score_map(dataflow_item)

        fused_by_doc: dict[str, tuple[str, float]] = {}
        for doc in sparse_items + dataflow_item:
            score_bm25 = sparse_score_map.get(doc, 0.0)
            score_dfg = dataflow_score_map.get(doc, 0.0)
            final_score = alpha * score_bm25 + (1 - alpha) * score_dfg
            key = normalize_doc_key(doc)
            old = fused_by_doc.get(key)
            # Deduplicate by normalized text and keep the higher fused score.
            if old is None:
                fused_by_doc[key] = (doc, final_score)

        fused = list(fused_by_doc.values())

        if use_sorted:
            fused.sort(key=lambda x: x[1], reverse=True)
        else:
            random.shuffle(fused)

        top_docs = fused[: min(top_k, len(fused))]
        result_blocks.append(
            [
                ResultBlock(block_idx=idx, code_snippet=doc)
                for idx, (doc, _) in enumerate(top_docs)
            ]
        )

    logger.info(
        f"Rerank completed with alpha={alpha}, sorted={use_sorted}, top_k={top_k}. "
        f"output={settings.rerank.output_file}"
    )
    saving_result = RerankResultSaving(data_list=result_blocks)
    settings.rerank.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.rerank.output_file, "w", encoding="utf-8") as f:
        f.write(saving_result.model_dump_json(indent=4))

            

            


if __name__ == "__main__":
    main()