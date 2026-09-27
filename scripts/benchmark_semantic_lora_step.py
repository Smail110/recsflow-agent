"""Time one exact QLoRA microbatch to diagnose the local training execution path."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.semantic_lora_common import (
    canonical,
    load_config,
    read_jsonl,
    resolve,
    seen_schema_holdout,
    tokenize_training_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deterministic-kernels", choices=("true", "false"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    deterministic = args.deterministic_kernels == "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config["seeds"]["torch"])
    torch.cuda.manual_seed_all(config["seeds"]["torch"])
    base = config["base_model"]
    training = config["training"]
    snapshot = Path(snapshot_download(base["id"], revision=base["revision"], local_files_only=base["local_files_only"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    rows, _ = seen_schema_holdout(rows, config["dataset"]["seen_schema_eval_fraction"], config["seeds"]["split"])
    encoded = tokenize_training_row(rows[0], tokenizer, training["max_sequence_length"], training.get("prompt_mode", "production"))
    quantization = training["quantization"]
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quantization["load_in_4bit"],
            bnb_4bit_quant_type=quantization["type"],
            bnb_4bit_use_double_quant=quantization["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": training["gradient_checkpointing_use_reentrant"]},
    )
    lora = training["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            task_type=lora["task_type"],
            use_rslora=lora["use_rslora"],
        ),
    )
    batch = {key: torch.tensor([encoded[key]], dtype=torch.long, device="cuda") for key in ("input_ids", "attention_mask", "labels")}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = model(**batch).loss
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    print(
        canonical(
            {
                "deterministic_kernels": deterministic,
                "example_id": encoded["id"],
                "tokens": encoded["length"],
                "loss": float(loss.detach().cpu()),
                "seconds": seconds,
                "tokens_per_second": encoded["length"] / seconds,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
