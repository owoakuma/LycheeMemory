import argparse
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
from transformers import AutoTokenizer

from src.modeling_qwen2 import Qwen2ForCausalLM

def _read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        path = Path(args.text_file)
        if path.suffix.lower() == ".json":
            df = pd.read_json(path)
            row = df.iloc[args.row_idx]
            return str(row["context"])
        return path.read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise ValueError("Please provide --text_file or --text.")

def _read_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    if args.text_file and Path(args.text_file).suffix.lower() == ".json":
        df = pd.read_json(args.text_file)
        row = df.iloc[args.row_idx]
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, dict) and "question" in extra_info:
            return str(extra_info["question"])
    raise ValueError("Please provide --question, or use a parquet file with extra_info.question.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick test for Qwen2 generate(text_ids=...) memory path."
    )
    parser.add_argument("--model_path", type=str, default="lerverson/LycheeMemory-7B", help="Model path or HF repo id.")
    parser.add_argument("--text_file", type=str, default="test.json", help="Long context text file path.")
    parser.add_argument("--text", type=str, default="", help="Long context text content.")
    parser.add_argument("--question", type=str, default="", help="Question prompt.")
    parser.add_argument("--row_idx", type=int, default=0, help="Row index")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument("--stream_chunk_size", type=int, default=4096)
    parser.add_argument("--lychee_memory_ratio", type=int, default=4)
    parser.add_argument("--lychee_memory_window", type=int, default=2048)
    parser.add_argument("--lychee_memory_stride", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = _read_text(args)
    question = _read_question(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
        low_cpu_mem_usage=False,
        local_files_only=args.local_files_only,
    )
    model.memory.set(
        lychee_memory_ratio=[args.lychee_memory_ratio],
        lychee_memory_window=args.lychee_memory_window,
        lychee_memory_stride=args.lychee_memory_stride,
    )
    meta_params = [name for name, param in model.named_parameters() if param.is_meta]
    if meta_params:
        preview = ", ".join(meta_params[:8])
        more = "" if len(meta_params) <= 8 else f" ... (+{len(meta_params) - 8} more)"
        raise RuntimeError(
            "Model still contains meta tensors after loading. "
            f"Examples: {preview}{more}. "
            "Please verify checkpoint completeness and loading options."
        )
    model = model.to(args.device).eval()

    text_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    question_inputs = tokenizer(question, return_tensors="pt", add_special_tokens=False)

    text_ids = text_inputs["input_ids"].to(args.device)
    text_attention_mask = text_inputs["attention_mask"].to(args.device)
    question_ids = question_inputs["input_ids"].to(args.device)
    question_attention_mask = question_inputs["attention_mask"].to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=question_ids,
            attention_mask=question_attention_mask,
            text_ids=text_ids,
            text_attention_mask=text_attention_mask,
            stream_chunk_size=args.stream_chunk_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            memory_mode="recurrent",
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = question_ids.shape[1]
    answer_ids = outputs[:, prompt_len:]
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()

    print("=== Input Stats ===")
    print(f"text_tokens: {text_ids.shape[1]}")
    print(f"question_tokens: {question_ids.shape[1]}")
    print(f"row_idx: {args.row_idx}")
    print(f"question: {question}")
    print(f"stream_chunk_size: {args.stream_chunk_size}")
    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
