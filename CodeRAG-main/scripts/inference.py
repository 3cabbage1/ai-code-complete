from loguru import logger
from coderag.config import settings
import json
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
class StopOnNewlineCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 获取换行符的token ID
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后生成的token是否是换行符或EOS
        last_token = input_ids[0, -1].item()
        if last_token == self.eos_token_id:
            return True
        if len(self.newline_ids) == 1 and last_token == self.newline_ids[0]:
            return True
        return False

def main():
    with open(settings.inference.use_prompt_file, "r", encoding="utf-8") as f:
        prompt_list = json.load(f)

    if settings.sample_n is not None:
        prompt_list = prompt_list[:settings.sample_n]

    def _normalize_prompt(p: Any) -> str:
        """
        The pipeline expects a list of prompt strings, but some users may save
        chat-style prompts. We normalize them to a plain string for local HF generation.
        """
        if isinstance(p, str):
            return p
        if isinstance(p, dict):
            # common shapes: {"content": "..."} or {"text": "..."}
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

    model_name = "Salesforce/codegen-350M-mono"
    logger.info(f"Loading local HF model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if not use_cuda:
        model.to(device)
    model.eval()

    results: list[str] = []
    error_count = 0
    error_ids: list[int] = []
# ####test
#     count=0
# ####test

    # 创建停止标准
    stopping_criteria = StoppingCriteriaList([StopOnNewlineCriteria(tokenizer)])

    for idx, raw_prompt in enumerate(prompt_list):
# ####test        
#         count+=1
#         if count>60:
#             break
# ####test

        task_id = f"Task {idx + 1}/{len(prompt_list)}"
        prompt = _normalize_prompt(raw_prompt)
# ####test
#         if count==1:
#             print(f"prompt: {prompt}\n")
# ####test
        try:
            logger.info(f"{task_id}: generating (prompt_prefix={prompt[:50]!r})")
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=settings.inference.max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,  # 使用正确的停止标准
                )
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = output_ids[0][prompt_len:]
            result_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(result_text)
        except Exception as e:
            error_count += 1
            error_ids.append(idx)
            logger.error(f"{task_id}: inference failed: {e}")
            results.append("")

    logger.info(f"Total errors: {error_count}")
    logger.info(f"Error IDs: {error_ids}")

    settings.inference.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.inference.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info("Inference completed and results saved.")

if __name__ == "__main__":
    main()