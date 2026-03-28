from pathlib import Path
from typing import List, Tuple
from loguru import logger
from coderag.config import settings
import json
from coderag.benchmark.methods import load_benchmark
from coderag.retrieve.dataflow_retrieve import DataflowRetriever
from coderag.retrieve.common import RetrieveResultItem, RetrieveResult
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval


def main():
    retrieve_settings = settings.retrieve

    logger.info(f"loading query file {retrieve_settings.use_query_file}")
    with open(retrieve_settings.use_query_file, "r") as f:
        querys: List[str] = json.load(f)

    logger.info("Retrieving...")

    benchmark = load_benchmark()
    sample_n = settings.sample_n
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]
        querys = querys[:sample_n]
    benchmark_count = len(benchmark.data_list)

    dataflow_retrieval: List[List[str]]= []
    if retrieve_settings.dataflow.enable:
        tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

        logger.info("Using dataflow retrieval...")
        dataflow_retriever = DataflowRetriever(
            projs_dir=settings.benchmark.repos_path,
            cache_dir=settings.retrieve.dataflow.graph_cache_dir,
            use_cache=settings.retrieve.dataflow.graph_use_cache,
        )
        count = 0
        for benchmark_item in benchmark.data_list:
            repo_path = benchmark.get_repo(benchmark_item.repo_name).repo_path
            result = dataflow_retriever.retrieve(
                project_name=benchmark_item.repo_name,
                fpath=benchmark_item.file_path,
                source_code=benchmark_item.code_context,
                calc_truncated=lambda x: merge_retrieval(
                    retrieval_infos=x,
                    source_code_prefix=f"# {"/".join(benchmark_item.deduped_path_list)}",
                    source_code=benchmark_item.code_context,
                    tokenizer=tokenizer,
                )[1],
            )
            dataflow_retrieval.append(result)
            logger.debug(
                f"Dataflow retrieving {count}/{benchmark_count} done, task name: {benchmark_item.task_name}, query starts with: {benchmark_item.code_context[:30]}"
            )
            count += 1

    result = RetrieveResult(
        data_list=[]
    )
    for i in range(len(querys)):
        if settings.retrieve.dataflow.enable:
            dataflow_it = dataflow_retrieval[i]
        else:
            dataflow_it = None
        result.data_list.append(
            RetrieveResultItem(
                dataflow=dataflow_it
            )
        )
    retrieve_settings.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(retrieve_settings.output_file, "w" ,encoding='utf-8') as f:
        f.write(result.model_dump_json(indent=4))

if __name__ == "__main__":
    main()
