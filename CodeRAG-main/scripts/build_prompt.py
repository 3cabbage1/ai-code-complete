from loguru import logger
from coderag.config import settings
from coderag.benchmark import load_benchmark
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval
from coderag.retrieve.common import RetrieveResult
from operator import itemgetter
import json

def main():
    sample_n = settings.sample_n
    benchmark = load_benchmark()
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]

    retrieve_result = None
    if settings.build_prompt.use_retrieval and settings.build_prompt.use_retrieve_file is not None:
        with open(settings.build_prompt.use_retrieve_file, "r", encoding='utf-8') as f:
            retrieve_result = RetrieveResult.model_validate_json(f.read())
        if settings.build_prompt.retrieve_data_indices_path is not None:
            with open(settings.build_prompt.retrieve_data_indices_path, "r", encoding='utf-8') as f:
                retrieve_data_indices: list[int] = json.load(f)
            retrieve_result.data_list = itemgetter(*retrieve_data_indices)(retrieve_result.data_list)
        logger.info(f"use retrieval result in {settings.build_prompt.use_retrieve_file}")
    else:
        logger.info(f"without retrieval. zero shot.")



    result_prompt: list[str] = []

    total = len(benchmark.data_list)
    tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

    for i in range(total):
        benchmark_item = benchmark.data_list[i]
        retrieve_prompts: list[str] = []
        if retrieve_result is not None:
            retrieve_item = retrieve_result.data_list[i]
            dataflow_items = retrieve_item.dataflow or []
            for item in dataflow_items:
                retrieve_prompts.append(f"{item.replace("'''", '"""')}")

        source_code_prefix = f"# {"/".join(benchmark_item.deduped_path_list)}"

        user_prompt, retrieval_truncated, source_code_truncated = merge_retrieval(
            retrieval_infos=retrieve_prompts,
            source_code=benchmark_item.code_context,
            source_code_prefix=source_code_prefix,
            tokenizer=tokenizer,
        )
        result_prompt.append(user_prompt)
        logger.debug(
            f"Building prompt {i + 1}/{total} done, task: {benchmark_item.task_name}, result starts with: {user_prompt[:30]}"
        )

    settings.build_prompt.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.build_prompt.output_file, "w", encoding='utf-8') as f:
        json.dump(result_prompt, f, indent=4)

    
if __name__ == "__main__":
    main()
