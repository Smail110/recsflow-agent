"""Validate QLoRA learning, changed weights and adapter save/reload on a tiny set."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.semantic_lora_common import (
    ROOT,
    canonical,
    file_sha256,
    load_config,
    read_jsonl,
    resolve,
    sha256_bytes,
    stable_key,
    training_execution_sha256,
)


def _emit(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def _adapter_file(path: Path) -> Path:
    candidates = sorted(path.glob("adapter_model.*"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one adapter weight file in {path}, got {candidates}")
    return candidates[0]


def _seed(config: dict[str, Any], torch: Any) -> None:
    seeds = config["seeds"]
    os.environ["PYTHONHASHSEED"] = str(seeds["python"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    torch.cuda.manual_seed_all(seeds["torch"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_base(config: dict[str, Any], snapshot: Path, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = config["training"]["quantization"]
    return AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=config["base_model"]["trust_remote_code"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["type"],
            bnb_4bit_use_double_quant=quant["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )


def _batch(encoded: dict[str, Any], torch: Any) -> dict[str, Any]:
    return {key: torch.tensor([encoded[key]], dtype=torch.long, device="cuda") for key in ("input_ids", "attention_mask", "labels")}


def _reproduction(model: Any, selected: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    target_tokens = 0
    correct_tokens = 0
    exact_sequences = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for encoded in selected:
            batch = _batch(encoded, torch)
            output = model(**batch)
            shifted_labels = batch["labels"][:, 1:]
            mask = shifted_labels.ne(-100)
            predictions = output.logits[:, :-1].argmax(dim=-1)
            count = int(mask.sum().item())
            correct = int((predictions.eq(shifted_labels) & mask).sum().item())
            total_loss += float(output.loss.detach().cpu()) * count
            target_tokens += count
            correct_tokens += correct
            exact_sequences += int(correct == count)
    torch.cuda.synchronize()
    return {
        "target_nll": total_loss / target_tokens,
        "target_token_accuracy": correct_tokens / target_tokens,
        "exact_sequences": exact_sequences,
        "examples": len(selected),
        "target_tokens": target_tokens,
        "seconds": time.perf_counter() - started,
    }


def _child(config_path: Path, output_path: Path) -> None:
    import bitsandbytes as bnb
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    from scripts.semantic_lora_common import tokenize_training_row

    config = load_config(config_path)
    tiny = config["tiny_overfit"]
    training = config["training"]
    _seed(config, torch)
    snapshot = Path(
        snapshot_download(
            config["base_model"]["id"],
            revision=config["base_model"]["revision"],
            local_files_only=config["base_model"]["local_files_only"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    encoded = [
        tokenize_training_row(
            row,
            tokenizer,
            training["max_sequence_length"],
            training["prompt_mode"],
        )
        for row in rows
    ]
    selected = sorted(
        encoded,
        key=lambda item: (
            item["length"],
            stable_key(config["seeds"]["tiny_selection"], item["id"]),
        ),
    )[: tiny["examples"]]
    _emit(
        "selection",
        ids=[item["id"] for item in selected],
        lengths=[item["length"] for item in selected],
    )

    model = _load_base(config, snapshot, torch)
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
    trainable, total = model.get_nb_trainable_parameters()
    output_dir = resolve(tiny["output_dir"]).resolve()
    allowed_root = (ROOT / "artifacts" / "lora" / "semantic_extraction_v2").resolve()
    if not output_dir.is_relative_to(allowed_root) or output_dir == allowed_root:
        raise ValueError(f"unsafe tiny output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    initial_dir = output_dir / "initial-adapter"
    final_dir = output_dir / "final-adapter"
    reloaded_dir = output_dir / "reloaded-adapter"
    model.save_pretrained(initial_dir, safe_serialization=True)
    initial_weights = _adapter_file(initial_dir)
    initial_sha = file_sha256(initial_weights)
    before = _reproduction(model, selected, torch)
    _emit("before", metrics=before, allocated_bytes=torch.cuda.memory_allocated())

    optimizer = bnb.optim.PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=tiny["learning_rate"],
    )
    losses: list[float] = []
    step_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(tiny["max_steps"]):
        started = time.perf_counter()
        output = model(**_batch(selected[step % len(selected)], torch))
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        loss = float(output.loss.detach().cpu())
        losses.append(loss)
        step_seconds.append(elapsed)
        _emit("train_step", step=step + 1, loss=loss, seconds=elapsed)
        if elapsed > tiny["step_timeout_seconds"]:
            raise TimeoutError(f"step {step + 1} exceeded {tiny['step_timeout_seconds']} seconds")

    after = _reproduction(model, selected, torch)
    model.save_pretrained(final_dir, safe_serialization=True)
    final_weights = _adapter_file(final_dir)
    final_sha = file_sha256(final_weights)
    weights_changed = initial_sha != final_sha
    _emit("after", metrics=after, weights_changed=weights_changed)

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(_load_base(config, snapshot, torch), final_dir)
    reloaded_metrics = _reproduction(reloaded, selected, torch)
    reloaded.save_pretrained(reloaded_dir, safe_serialization=True)
    reloaded_sha = file_sha256(_adapter_file(reloaded_dir))
    reload_nll_delta = abs(reloaded_metrics["target_nll"] - after["target_nll"])
    reload_accuracy_delta = abs(reloaded_metrics["target_token_accuracy"] - after["target_token_accuracy"])
    reload_equivalent = (
        reload_nll_delta <= tiny["reload_nll_absolute_tolerance"]
        and reload_accuracy_delta <= tiny["reload_token_accuracy_absolute_tolerance"]
        and reloaded_metrics["exact_sequences"] == after["exact_sequences"]
        and reloaded_sha == final_sha
    )
    loss_relative_drop = (before["target_nll"] - after["target_nll"]) / before["target_nll"]
    accuracy_gain = after["target_token_accuracy"] - before["target_token_accuracy"]
    gates = {
        "real_steps_completed": len(losses) == tiny["max_steps"],
        "loss_relative_drop": loss_relative_drop >= tiny["required_loss_relative_drop"],
        "target_token_accuracy_gain": (accuracy_gain >= tiny["required_target_token_accuracy_gain"]),
        "exact_sequence_gain": (after["exact_sequences"] - before["exact_sequences"] >= tiny["required_exact_sequence_gain"]),
        "adapter_weights_changed": weights_changed,
        "save_reload_equivalent": reload_equivalent,
        "practical_step_speed": max(step_seconds) <= tiny["step_timeout_seconds"],
    }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "config_sha256": file_sha256(config_path),
        "training_execution_sha256": training_execution_sha256(config),
        "dataset_train_sha256": config["dataset"]["train"]["sha256"],
        "selected_ids": [item["id"] for item in selected],
        "selected_lengths": [item["length"] for item in selected],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "steps": len(losses),
        "losses": losses,
        "step_seconds": step_seconds,
        "before": before,
        "after": after,
        "reloaded": reloaded_metrics,
        "loss_relative_drop": loss_relative_drop,
        "target_token_accuracy_gain": accuracy_gain,
        "adapter_weights": {
            "initial_sha256": initial_sha,
            "final_sha256": final_sha,
            "reloaded_sha256": reloaded_sha,
        },
        "reload_nll_absolute_delta": reload_nll_delta,
        "reload_token_accuracy_absolute_delta": reload_accuracy_delta,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "gates": gates,
        "final_holdout_used": False,
    }
    result["report_sha256"] = sha256_bytes(canonical(result).encode())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit("complete", status=result["status"], report_sha256=result["report_sha256"])
    if result["status"] != "PASS":
        raise SystemExit(2)


def _events(output: str | bytes | None) -> list[dict[str, Any]]:
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
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        _child(args.config, args.output)
        return
    config = load_config(args.config)
    timeout = config["tiny_overfit"]["total_timeout_seconds"]
    command = [
        sys.executable,
        "-m",
        "scripts.tiny_overfit_semantic_lora",
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--_child",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        diagnostic = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "TIMEOUT",
            "timeout_seconds": timeout,
            "wall_seconds": time.perf_counter() - started,
            "config_sha256": file_sha256(args.config),
            "training_execution_sha256": training_execution_sha256(config),
            "events": _events(exc.stdout),
            "stderr_tail": str(exc.stderr or "")[-4000:],
            "final_holdout_used": False,
        }
        diagnostic["report_sha256"] = sha256_bytes(canonical(diagnostic).encode())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(canonical(diagnostic))
        raise SystemExit(2) from exc
    print(completed.stdout, end="")
    if completed.returncode:
        if not args.output.exists():
            diagnostic = {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "status": "ERROR",
                "wall_seconds": time.perf_counter() - started,
                "config_sha256": file_sha256(args.config),
                "training_execution_sha256": training_execution_sha256(config),
                "events": _events(completed.stdout),
                "stderr_tail": completed.stderr[-4000:],
                "returncode": completed.returncode,
                "final_holdout_used": False,
            }
            diagnostic["report_sha256"] = sha256_bytes(canonical(diagnostic).encode())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
