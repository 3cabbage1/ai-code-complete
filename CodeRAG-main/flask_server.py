import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request as flask_request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from coderag.config import settings
from coderag.build_query.build_query import build_query_by_last_k_lines
from coderag.retrieve.dataflow_retrieve import DataflowRetriever
from coderag.retrieve.methods import retrieve_code_snippets_sparse
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval
# 直接使用 projectParser 解析当前项目，绕过 generate_context_graph
from coderag.static_analysis.data_flow.preprocess import projectParser



app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE: dict[str, Any] = {
    "workspace_path": None,
    "project_name": None,
    "retriever": None,
    "cache_dir": None,
    "lm": None,
    "lm_tokenizer": None,
    "lm_device": None,
    "stopping_criteria": None,
}


def _calc_truncated_factory(max_chars: int) -> Callable[[list[str]], bool]:
    def calc_truncated(prompt_list: list[str]) -> bool:
        total = sum(len(x) for x in prompt_list)
        return total > max_chars

    return calc_truncated


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


def ensure_retriever(workspace_path: str, force_rebuild: bool = False) -> DataflowRetriever:
    workspace = Path(workspace_path).resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")
    # workspace:当前打开的目录
    project_name = workspace.name 
    projs_dir = workspace.parent #当前目录父目录
    cache_dir = workspace / ".ai-code-complete" / "dataflow-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    retriever = STATE.get("retriever")
    if (
        retriever is not None
        and STATE.get("workspace_path") == str(workspace)
        and not force_rebuild
    ):
        logger.info("=" * 80)
        logger.info("USING CACHED RETRIEVER")
        logger.info(f"Workspace path: {STATE.get('workspace_path')}")
        logger.info(f"Project name: {STATE.get('project_name')}")
        logger.info("=" * 80)
        return retriever

    cache_has_project = (cache_dir / f"{project_name}.json").exists()
    if force_rebuild or not cache_has_project:
        # Only generate current project graph, avoid scanning all siblings.
        if force_rebuild:
            logger.info(f"Cache directory: {cache_dir}")
            for p in cache_dir.iterdir():
                try:
                    p.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {p.name}: {e}")
            logger.info("Cache directory cleared")
            logger.info("=" * 80)
        
        logger.info("=" * 80)
        logger.info("GENERATING CONTEXT GRAPH")
        logger.info(f"Project name: {project_name}")
        logger.info(f"Workspace path: {workspace}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info("=" * 80)
        
        
        # project_parser = projectParser()
        # logger.info(f"Parsing project directory: {workspace}")
        # info = project_parser.parse_dir(str(workspace))
        # logger.info("=" * 80)
        # logger.info(f"info: {info}")
        # logger.info(f"Parsed info keys: {list(info.keys()) if info else 'empty'}")
        project_parser = projectParser()
        dir_path=workspace
        if dir_path.is_dir():
            content = list(dir_path.iterdir())
            if len(content) > 1:
                info = project_parser.parse_dir(str(dir_path))
                logger.info("#" * 80)
                logger.info(f"dir_path: {str(dir_path)}")
                logger.info(f"len(content) > 1:\ninfo: {info}")
                logger.info("#" * 80)
            else:
                # package/package-version/
                dist_path = content[0]
                info = project_parser.parse_dir(str(dist_path))
            cache_file = cache_dir / f"{project_name}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(info, f)



        # 打印生成的 JSON 内容（限制大小）
        json_content = json.dumps(info, ensure_ascii=False, indent=2)
        max_json_length = 10000  # 限制日志长度
        if len(json_content) > max_json_length:
            logger.info(f"Generated JSON content (truncated): {json_content[:max_json_length]}...")
            logger.info(f"Total JSON size: {len(json_content)} characters")
        else:
            logger.info(f"Generated JSON content: {json_content}")
        
        # # 保存到缓存
        # cache_file = cache_dir / f"{project_name}.json"
        # with open(cache_file, 'w', encoding='utf-8') as f:
        #     json.dump(info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Context graph saved to: {cache_file}")
        logger.info(f"File size: {cache_file.stat().st_size if cache_file.exists() else 0} bytes")
        
        logger.info("=" * 80)
        logger.info("CONTEXT GRAPH GENERATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Cache file: {cache_file}")
        logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info("USING CACHED CONTEXT GRAPH")
        logger.info("=" * 80)
        logger.info(f"Cache file: {cache_dir / f'{project_name}.json'}")
        logger.info("=" * 80)

    retriever = DataflowRetriever(
        projs_dir=projs_dir,
        cache_dir=cache_dir,
        use_cache=True,
    )

    STATE["workspace_path"] = str(workspace)
    STATE["project_name"] = project_name
    STATE["retriever"] = retriever
    STATE["cache_dir"] = str(cache_dir)
    
    logger.info("=" * 80)
    logger.info("RETRIEVER CREATED AND CACHED")
    logger.info(f"Workspace path: {STATE.get('workspace_path')}")
    logger.info(f"Project name: {STATE.get('project_name')}")
    logger.info(f"Cache directory: {STATE.get('cache_dir')}")
    logger.info("=" * 80)
    
    return retriever


class StopOnNewlineCriteria(StoppingCriteria):
    def __init__(self, tokenizer, min_new_tokens: int = 2):
        self.tokenizer = tokenizer
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        self.eos_token_id = tokenizer.eos_token_id
        self.min_new_tokens = min_new_tokens
        self._start_len: int | None = None

    def __call__(self, input_ids, scores, **kwargs):
        # StoppingCriteria only sees the full input_ids (prompt+generated).
        # We infer newly-generated token count from the first call.
        if self._start_len is None:
            self._start_len = input_ids.shape[1]
        generated_tokens = input_ids.shape[1] - self._start_len

        last_token = input_ids[0, -1].item()
        if last_token == self.eos_token_id:
            return True
        if generated_tokens >= self.min_new_tokens and len(self.newline_ids) == 1 and last_token == self.newline_ids[0]:
            return True
        return False


def ensure_lm_loaded() -> tuple[Any, Any, Any, Any]:
    """
    Load local HF model once, following CodeRAG-main/scripts/inference.py behavior.
    """
    if STATE.get("lm") is not None and STATE.get("lm_tokenizer") is not None:
        return STATE["lm"], STATE["lm_tokenizer"], STATE["lm_device"], STATE["stopping_criteria"]

    logger.info("=" * 80)
    logger.info("STARTING MODEL LOADING...")
    logger.info("=" * 80)

    model_name = "Salesforce/codegen-350M-mono"
    logger.info(f"Model name: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    logger.info("=" * 80)
    logger.info(f"DEVICE:{device.type}")
    logger.info("=" * 80)
    dtype = torch.float16 if use_cuda else torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    
    # 添加模型量化以提高推理速度
    if use_cuda:
        # 对于GPU，使用半精度推理
        model.half()
    else:
        # 对于CPU，使用INT8量化
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    model.eval()
    if not use_cuda:
        model.to(device)

    stopping_criteria = StoppingCriteriaList([StopOnNewlineCriteria(tokenizer)])

    STATE["lm"] = model
    STATE["lm_tokenizer"] = tokenizer
    STATE["lm_device"] = device
    STATE["stopping_criteria"] = stopping_criteria

    logger.info("=" * 80)
    logger.info("MODEL LOADING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device.type}")
    logger.info(f"Data type: {dtype}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("=" * 80)

    return model, tokenizer, device, stopping_criteria


def normalize_prompt(p: Any) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in ("content", "text", "prompt"):
            v = p.get(k)
            if isinstance(v, str):
                return v
        return json.dumps(p, ensure_ascii=False)
    if isinstance(p, list):
        parts: list[str] = []
        for it in p:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict) and isinstance(it.get("content"), str):
                parts.append(it["content"])
            else:
                parts.append(str(it))
        return "\n".join(parts)
    return str(p)


@torch.no_grad()
def local_infer_completion(prompt: str, max_tokens: int | None = None) -> str:
    start_time = time.time()
    model, tokenizer, device, stopping_criteria = ensure_lm_loaded()
    normalized = normalize_prompt(prompt)

    inputs = tokenizer(normalized, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new_tokens = max_tokens if max_tokens is not None else settings.inference.max_tokens

    # Use a fresh stopping criteria instance per request so it can reset state.
    stopping_criteria = StoppingCriteriaList([StopOnNewlineCriteria(tokenizer, min_new_tokens=2)])

    # 优化推理参数以提高速度
    output_ids = model.generate(
        **inputs,
        max_new_tokens=min(max_new_tokens, 1028),  # 限制生成的token数量
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
        # 添加额外的优化参数
        use_cache=True,  # 启用缓存
        # 对于某些模型，添加以下参数可能会提高速度
        # num_return_sequences=1,
        # top_k=50,
        # top_p=0.95,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][prompt_len:]
    result_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    end_time = time.time()
    logger.info(f"Inference completed in {end_time - start_time:.2f} seconds using {device.type}")
    return result_text


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True, "workspace_path": STATE.get("workspace_path")})


@app.post("/index")
def index_workspace() -> Any:
    payload = flask_request.get_json(silent=True) or {}
    workspace_path = payload.get("workspace_path")
    force_rebuild = bool(payload.get("force_rebuild", False))
    if not workspace_path:
        return jsonify({"ok": False, "error": "workspace_path is required"}), 400

    logger.info("=" * 80)
    logger.info("INDEX WORKSPACE REQUEST RECEIVED")
    logger.info(f"Workspace path: {workspace_path}")
    logger.info(f"Force rebuild: {force_rebuild}")
    logger.info("=" * 80)

    try:
        logger.info("Starting to ensure retriever...")
        retriever = ensure_retriever(workspace_path, force_rebuild=force_rebuild)
        
        logger.info("=" * 80)
        logger.info("INDEX WORKSPACE COMPLETED SUCCESSFULLY")
        logger.info(f"Cache directory: {STATE.get('cache_dir')}")
        logger.info("=" * 80)
        
        return jsonify(
            {
                "ok": True,
                "workspace_path": STATE.get("workspace_path"),
                "project_name": STATE.get("project_name"),
                "cache_dir": STATE.get('cache_dir'),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("=" * 80)
        logger.info("INDEX WORKSPACE FAILED")
        logger.info(f"Error: {str(exc)}")
        logger.info("=" * 80)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/suggest")
def suggest() -> Any:
    payload = flask_request.get_json(silent=True) or {}
    workspace_path = payload.get("workspace_path")
    file_path = payload.get("file_path")
    source_code = payload.get("source_code")
    max_chars = int(payload.get("max_context_chars", 10000))
    enable_completion = bool(payload.get("enable_completion", True))
    max_tokens = int(payload.get("max_tokens", settings.inference.max_tokens))

    if not workspace_path or not file_path or source_code is None:
        return jsonify({"ok": False, "error": "workspace_path, file_path, source_code are required"}), 400

    try:
        retriever = ensure_retriever(workspace_path, force_rebuild=False)
        project_name = STATE.get("project_name")
        file_path_abs = str(file_path)

        # 0) 读取整文件源码，用于构建 dataflow 图
        try:
            with open(file_path_abs, "r", encoding="utf-8") as f:
                full_source = f.read()
        except UnicodeDecodeError:
            with open(file_path_abs, "r", encoding="utf-8", errors="ignore") as f:
                full_source = f.read()

        # 1) Build query from *context code* (靠近光标的前后片段)，更贴近 CodeRAG 的 code_context 语义
        # VS Code 传来的 source_code 是「光标前代码」，这里在其基础上做 last_k 裁剪。
        query_list = build_query_by_last_k_lines([source_code], k=settings.query.lask_k)
        query = query_list[0] if query_list else source_code
        
        logger.info("=" * 80)
        logger.info("QUERY FOR RETRIEVAL:")
        logger.info("=" * 80)
        logger.info(query)
        logger.info("=" * 80)

        # 2) Dataflow retrieval context (CodeRAG-main/scripts/retrieve.py)
        #    - dataflow 图用整文件 full_source 构建（更容易看到 import / 跨文件关系）
        #    - query 只作为「当前问题位置的上下文」参与提示词构建
        rel_file = os.path.relpath(file_path_abs, workspace_path)
        rel_file_norm = rel_file.replace("\\", "/")
        source_code_prefix = f"# {rel_file_norm}"
        tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

        logger.info("=" * 80)
        logger.info("DATAFLOW RETRIEVAL DEBUG INFO:")
        logger.info("=" * 80)
        logger.info(f"Project name: {project_name}")
        logger.info(f"File path (abs): {file_path_abs}")
        logger.info(f"File path (rel): {rel_file}")
        logger.info(f"File path (norm): {rel_file_norm}")
        logger.info(f"Source code length: {len(query)} chars")
        logger.info("=" * 80)

        def calc_truncated(retrieval_infos: list[str]) -> bool:
            # merge_retrieval 返回: (prompt, source_code_truncated, retrieval_truncated)
            # 这里只关心「检索信息是否被截断」，与 CodeRAG scripts 保持一致。
            _prompt, _source_truncated, retrieval_truncated = merge_retrieval(
                retrieval_infos=retrieval_infos,
                source_code_prefix=source_code_prefix,
                source_code=query,
                tokenizer=tokenizer,
            )
            return retrieval_truncated

        # 并行执行 Dataflow 和 BM25 检索
        logger.info("Starting parallel retrieval (Dataflow + BM25)...")
        
        def run_dataflow_retrieval():
            logger.info("Starting dataflow retrieval...")
            result = retriever.retrieve(
                project_name=project_name,
                fpath=Path(file_path_abs),
                # 使用整文件源码构建 DFG，保证能看到文件顶部的 import / 类 / 函数定义
                source_code=full_source,
                calc_truncated=calc_truncated,
            )
            logger.info(f"Dataflow retrieval completed. Retrieved {len(result or [])} contexts.")
            return result
        
        def run_bm25_retrieval():
            logger.info("Starting BM25 retrieval...")
            result = retrieve_code_snippets_sparse(
                repo_path=Path(workspace_path),
                query=query,
                k_func=5,
                k_var=5,
                method="bm25",
                exclude_path=Path(file_path_abs),
                tokenizer=lambda text: text.split()
            )
            logger.info(f"BM25 retrieval completed. Retrieved {len(result or [])} contexts.")
            return result
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dataflow = executor.submit(run_dataflow_retrieval)
            future_bm25 = executor.submit(run_bm25_retrieval)
            
            # 获取结果
            prompt_list = future_dataflow.result()
            bm25_results = future_bm25.result()
        
        logger.info("Parallel retrieval completed.")

        logger.info("=" * 80)
        logger.info(f"RETRIEVED CONTEXTS (Dataflow: {len(prompt_list or [])}, BM25: {len(bm25_results or [])}):")
        logger.info("=" * 80)
        for idx, ctx in enumerate(prompt_list or [], 1):
            logger.info(f"--- Dataflow Context {idx} ---")
            logger.info(ctx)
        for idx, ctx in enumerate(bm25_results or [], 1):
            logger.info(f"--- BM25 Context {idx} ---")
            logger.info(ctx)
        logger.info("=" * 80)

        # 4) Rerank and fuse results
        logger.info("Starting reranking and fusing results...")
        alpha = 0.5  # Weight for BM25, (1-alpha) for dataflow
        top_k = 10  # Top k results to keep

        # Build score maps
        bm25_score_map = build_norm_score_map(bm25_results or [])
        dataflow_score_map = build_norm_score_map(prompt_list or [])

        # Fuse results
        fused_by_doc: dict[str, tuple[str, float]] = {}
        for doc in (bm25_results or []) + (prompt_list or []):
            score_bm25 = bm25_score_map.get(doc, 0.0)
            score_dataflow = dataflow_score_map.get(doc, 0.0)
            final_score = alpha * score_bm25 + (1 - alpha) * score_dataflow
            key = normalize_doc_key(doc)
            old = fused_by_doc.get(key)
            # Deduplicate by normalized text and keep the higher fused score
            if old is None or final_score > old[1]:
                fused_by_doc[key] = (doc, final_score)

        # Sort by score
        fused = list(fused_by_doc.values())
        fused.sort(key=lambda x: x[1], reverse=True)

        # Get top k results
        top_docs = fused[:min(top_k, len(fused))]
        reranked_results = [doc for doc, _ in top_docs]

        logger.info(f"Reranking completed. Kept top {len(reranked_results)} results.")
        logger.info("=" * 80)
        logger.info("RERANKED CONTEXTS:")
        logger.info("=" * 80)
        for idx, ctx in enumerate(reranked_results, 1):
            logger.info(f"--- Reranked Context {idx} ---")
            logger.info(ctx)
        logger.info("=" * 80)

        # 3) Build final prompt (CodeRAG-main/scripts/build_prompt.py)
        retrieval_prompts: list[str] = []
        for item in reranked_results:
            retrieval_prompts.append(item.replace("'''", '"""'))

        user_prompt, _retrieval_truncated, _source_code_truncated = merge_retrieval(
            retrieval_infos=retrieval_prompts,
            source_code_prefix=source_code_prefix,
            source_code=query,
            tokenizer=tokenizer,
        )

        logger.info("=" * 80)
        logger.info("BUILT PROMPT FOR INFERENCE:")
        logger.info("=" * 80)
        logger.info(user_prompt)
        logger.info("=" * 80)

        # 4) Local inference (CodeRAG-main/scripts/inference.py)
        completion = ""
        if enable_completion:
            completion = local_infer_completion(user_prompt, max_tokens=max_tokens)

        logger.info("=" * 80)
        logger.info("COMPLETION CODE:")
        logger.info("=" * 80)
        logger.info(completion or "(empty)")
        logger.info("=" * 80)

        return jsonify(
            {
                "ok": True,
                "project_name": project_name,
                "retrieved_contexts": retrieval_prompts,
                "completion": completion,
                "completion_chars": len(completion or ""),
                "retrieved_contexts_count": len(retrieval_prompts or []),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeRAG Flask server for VS Code completion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("FLASK SERVER STARTING...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info("=" * 80)
    
    logger.info("Preloading inference model...")
    ensure_lm_loaded()
    
    logger.info("=" * 80)
    logger.info("STARTING FLASK SERVER...")
    logger.info("=" * 80)
    
    app.run(host=args.host, port=args.port)
