import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, request as flask_request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from coderag.config import settings
from coderag.build_query.build_query import build_query_by_last_k_lines
from coderag.retrieve.dataflow_retrieve import DataflowRetriever
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval
from coderag.static_analysis.data_flow.preprocess import generate_context_graph


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


def ensure_retriever(workspace_path: str, force_rebuild: bool = False) -> DataflowRetriever:
    workspace = Path(workspace_path).resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")

    project_name = workspace.name
    projs_dir = workspace.parent
    cache_dir = workspace / ".ai-code-complete" / "dataflow-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    retriever = STATE.get("retriever")
    if (
        retriever is not None
        and STATE.get("workspace_path") == str(workspace)
        and not force_rebuild
    ):
        return retriever

    cache_has_project = (cache_dir / f"{project_name}.json").exists()
    if force_rebuild or not cache_has_project:
        # Only generate current project graph, avoid scanning all siblings.
        if force_rebuild:
            for p in cache_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
        generate_context_graph(
            pkg_list=[project_name],
            ds_repo_dir=projs_dir,
            ds_graph_dir=cache_dir,
        )

    retriever = DataflowRetriever(
        projs_dir=projs_dir,
        cache_dir=cache_dir,
        use_cache=True,
    )

    STATE["workspace_path"] = str(workspace)
    STATE["project_name"] = project_name
    STATE["retriever"] = retriever
    STATE["cache_dir"] = str(cache_dir)
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
    model, tokenizer, device, stopping_criteria = ensure_lm_loaded()
    normalized = normalize_prompt(prompt)

    inputs = tokenizer(normalized, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new_tokens = max_tokens if max_tokens is not None else settings.inference.max_tokens

    # Use a fresh stopping criteria instance per request so it can reset state.
    stopping_criteria = StoppingCriteriaList([StopOnNewlineCriteria(tokenizer, min_new_tokens=2)])

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][prompt_len:]
    result_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
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

    try:
        ensure_retriever(workspace_path, force_rebuild=force_rebuild)
        return jsonify(
            {
                "ok": True,
                "workspace_path": STATE.get("workspace_path"),
                "project_name": STATE.get("project_name"),
                "cache_dir": STATE.get("cache_dir"),
            }
        )
    except Exception as exc:  # noqa: BLE001
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

        # 1) Build query from source code (CodeRAG-main/scripts/build_query.py)
        #    For interactive completion, we use last_k lines.
        query_list = build_query_by_last_k_lines([source_code], k=settings.query.lask_k)
        query = query_list[0] if query_list else source_code
        
        logger.info("=" * 80)
        logger.info("QUERY FOR RETRIEVAL:")
        logger.info("=" * 80)
        logger.info(query)
        logger.info("=" * 80)

        # 2) Dataflow retrieval context (CodeRAG-main/scripts/retrieve.py)
        #    The truncation decision uses merge_retrieval's token logic, same as retrieve.py.
        rel_file = os.path.relpath(file_path_abs, workspace_path)
        rel_file_norm = rel_file.replace("\\", "/")
        source_code_prefix = f"# {rel_file_norm}"
        tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

        def calc_truncated(retrieval_infos: list[str]) -> bool:
            # merge_retrieval returns: (prompt, source_code_truncated, retrieval_truncated)
            return merge_retrieval(
                retrieval_infos=retrieval_infos,
                source_code_prefix=source_code_prefix,
                source_code=query,
                tokenizer=tokenizer,
            )[1]

        prompt_list = retriever.retrieve(
            project_name=project_name,
            fpath=Path(file_path_abs),
            source_code=query,
            calc_truncated=calc_truncated,
        )

        logger.info("=" * 80)
        logger.info(f"RETRIEVED CONTEXTS (count: {len(prompt_list or [])}):")
        logger.info("=" * 80)
        for idx, ctx in enumerate(prompt_list or [], 1):
            logger.info(f"--- Context {idx} ---")
            logger.info(ctx)
        logger.info("=" * 80)

        # 3) Build final prompt (CodeRAG-main/scripts/build_prompt.py)
        retrieval_prompts: list[str] = []
        for item in (prompt_list or []):
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
