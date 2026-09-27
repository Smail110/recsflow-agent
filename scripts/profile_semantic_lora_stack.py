"""Profile dataset, tokenization and one QLoRA optimizer step with a hard timeout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    load_config,
    sha256_bytes,
    training_execution_sha256,
)


def _event(stage: str, seconds: float, **details: Any) -> None:
    print(json.dumps({"stage": stage, "seconds": seconds, **details}, sort_keys=True), flush=True)


def _child(
    config_path: Path,
    checkpointing: bool,
    deterministic: bool,
    prompt_mode_override: str | None,
    max_sequence_length_override: int | None,
) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import bitsandbytes as bnb
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from scripts.semantic_lora_common import read_jsonl, resolve, tokenize_training_row

    config = load_config(config_path)
    training = config["training"]
    started = time.perf_counter()
    train_rows = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    dev_rows = read_jsonl(resolve(config["dataset"]["dev"]["path"]))
    _event(
        "dataset_load",
        time.perf_counter() - started,
        train_rows=len(train_rows),
        dev_rows=len(dev_rows),
        gpu_allocated_bytes=0,
    )

    base = config["base_model"]
    started = time.perf_counter()
    snapshot = Path(snapshot_download(base["id"], revision=base["revision"], local_files_only=base["local_files_only"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    _event(
        "tokenizer_load",
        time.perf_counter() - started,
        snapshot=str(snapshot),
        gpu_allocated_bytes=0,
    )

    started = time.perf_counter()
    prompt_mode = prompt_mode_override or training.get("prompt_mode", "production")
    max_sequence_length = max_sequence_length_override or training["max_sequence_length"]
    encoded_train = [tokenize_training_row(row, tokenizer, max_sequence_length, prompt_mode) for row in train_rows]
    encoded_dev = [tokenize_training_row(row, tokenizer, max_sequence_length, prompt_mode) for row in dev_rows]
    selected = min(encoded_train, key=lambda item: (item["length"], item["id"]))
    _event(
        "tokenization",
        time.perf_counter() - started,
        examples=len(encoded_train) + len(encoded_dev),
        selected_id=selected["id"],
        selected_tokens=selected["length"],
        maximum_tokens=max(item["length"] for item in encoded_train + encoded_dev),
        prompt_mode=prompt_mode,
        max_sequence_length=max_sequence_length,
        gpu_allocated_bytes=0,
    )

    deterministic_started = time.perf_counter()
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config["seeds"]["torch"])
    torch.cuda.manual_seed_all(config["seeds"]["torch"])
    _event(
        "deterministic_settings",
        time.perf_counter() - deterministic_started,
        enabled=deterministic,
        gpu_allocated_bytes=torch.cuda.memory_allocated(),
    )

    quant = training["quantization"]
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["type"],
            bnb_4bit_use_double_quant=quant["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    _event(
        "quantized_model_load",
        time.perf_counter() - started,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        bf16_snapshot_bytes=sum(path.stat().st_size for path in snapshot.glob("*.safetensors")),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
    )

    started = time.perf_counter()
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": training["gradient_checkpointing_use_reentrant"]} if checkpointing else None,
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
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    _event(
        "checkpointing_and_lora_setup",
        time.perf_counter() - started,
        checkpointing=checkpointing,
        trainable_parameters=trainable,
        total_parameters=total,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
    )

    batch = {key: torch.tensor([selected[key]], dtype=torch.long, device="cuda") for key in ("input_ids", "attention_mask", "labels")}
    optimizer = bnb.optim.PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.get("tiny_overfit", training)["learning_rate"],
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(**batch)
    torch.cuda.synchronize()
    _event(
        "forward",
        time.perf_counter() - started,
        loss=float(output.loss.detach().cpu()),
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output.loss.backward()
    torch.cuda.synchronize()
    _event(
        "backward",
        time.perf_counter() - started,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    _event(
        "optimizer_step",
        time.perf_counter() - started,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    _event("complete", 0.0, status="PASS")


def _parse_events(output: str | bytes | None) -> list[dict[str, Any]]:
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    events = []
    for line in (output or "").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "stage" in item:
            events.append(item)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-checkpointing", choices=("true", "false"), required=True)
    parser.add_argument("--deterministic-kernels", choices=("true", "false"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--prompt-mode")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    checkpointing = args.gradient_checkpointing == "true"
    deterministic = args.deterministic_kernels == "true"
    if args._child:
        _child(
            args.config,
            checkpointing,
            deterministic,
            args.prompt_mode,
            args.max_sequence_length,
        )
        return
    command = [
        sys.executable,
        "-m",
        "scripts.profile_semantic_lora_stack",
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--gradient-checkpointing",
        args.gradient_checkpointing,
        "--deterministic-kernels",
        args.deterministic_kernels,
        "--timeout-seconds",
        str(args.timeout_seconds),
        *(["--prompt-mode", args.prompt_mode] if args.prompt_mode else []),
        *(["--max-sequence-length", str(args.max_sequence_length)] if args.max_sequence_length else []),
        "--_child",
    ]
    started = time.perf_counter()
    status = "ERROR"
    returncode = None
    stderr = ""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout_seconds, check=False)
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        status = "PASS" if returncode == 0 else "ERROR"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "TIMEOUT"
    loaded_config = load_config(args.config)
    execution_hash = training_execution_sha256(loaded_config) if "tiny_overfit" in loaded_config else None
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "wall_seconds": time.perf_counter() - started,
        "timeout_seconds": args.timeout_seconds,
        "gradient_checkpointing": checkpointing,
        "deterministic_kernels": deterministic,
        "prompt_mode_override": args.prompt_mode,
        "max_sequence_length_override": args.max_sequence_length,
        "config_sha256": file_sha256(args.config),
        "training_execution_sha256": execution_hash,
        "events": _parse_events(stdout),
        "returncode": returncode,
        "stderr_tail": str(stderr)[-4000:],
        "final_holdout_used": False,
    }
    report["report_sha256"] = sha256_bytes(canonical(report).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical(report))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
