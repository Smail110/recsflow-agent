"""Train the frozen schema-conditioned QLoRA pilot with assistant-only loss."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from scripts.evaluate_semantic_lora import verify_baseline
from scripts.semantic_lora_common import (
    canonical,
    file_sha256,
    latent_isolation_summary,
    load_config,
    read_jsonl,
    resolve,
    seen_schema_holdout,
    sha256_bytes,
    tokenize_training_row,
    training_execution_sha256,
    verify_frozen_protocol,
)


class TokenizedRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: value for key, value in self.rows[index].items() if key != "id"}


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _seed_everything(config: dict[str, Any]) -> None:
    seeds = config["seeds"]
    deterministic = config["training"]["full_determinism"]
    os.environ["PYTHONHASHSEED"] = str(seeds["python"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    torch.manual_seed(seeds["torch"])
    torch.cuda.manual_seed_all(seeds["torch"])
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    set_seed(seeds["trainer"], deterministic=deterministic)


def _verify_preflight(path: Path, config_hash: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    integrity = result.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(result).encode()):
        raise ValueError("preflight report integrity mismatch")
    if result["status"] != "PASS" or result["config_sha256"] != config_hash:
        raise ValueError("training requires PASS preflight for this exact config")
    result["report_sha256"] = integrity
    return result


def _artifact_files(path: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(item), "bytes": item.stat().st_size, "sha256": file_sha256(item)} for item in sorted(path.rglob("*")) if item.is_file()
    ]


def _verify_gate(
    path: Path,
    execution_hash: str,
    *,
    expected_checkpointing: bool | None = None,
    expected_deterministic: bool | None = None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    integrity = result.pop("report_sha256", None)
    if integrity != sha256_bytes(canonical(result).encode()):
        raise ValueError(f"gate report integrity mismatch: {path}")
    if result["status"] != "PASS":
        raise ValueError(f"gate is not PASS: {path}")
    if result.get("training_execution_sha256") != execution_hash:
        raise ValueError(f"gate execution config mismatch: {path}")
    if expected_checkpointing is not None and result.get("gradient_checkpointing") is not expected_checkpointing:
        raise ValueError(f"profile checkpointing mismatch: {path}")
    if expected_deterministic is not None and result.get("deterministic_kernels") is not expected_deterministic:
        raise ValueError(f"profile deterministic setting mismatch: {path}")
    result["report_sha256"] = integrity
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v2/preflight.json"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v2/evaluation/base.json"),
    )
    parser.add_argument(
        "--profile-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v2/profiles/checkpoint-on-nondeterministic.json"),
    )
    parser.add_argument(
        "--tiny-overfit-report",
        type=Path,
        default=Path("artifacts/lora/semantic_extraction_v2/tiny-overfit/report.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    evaluator_path = Path(__file__).with_name("evaluate_semantic_lora.py")
    protocol_seal = verify_frozen_protocol(args.config, config, evaluator_path)
    execution_hash = training_execution_sha256(config)
    profile = _verify_gate(
        args.profile_report,
        execution_hash,
        expected_checkpointing=config["training"]["gradient_checkpointing"],
        expected_deterministic=config["training"]["full_determinism"],
    )
    tiny = _verify_gate(args.tiny_overfit_report, execution_hash)
    preflight = _verify_preflight(args.preflight_report, config_hash)
    isolation = latent_isolation_summary(
        {split: read_jsonl(resolve(config["dataset"][split]["path"])) for split in ("train", "dev", "blind")}
    )
    if not isolation["passed"]:
        raise ValueError("template/schema isolation gate failed; v2 must not be trained")
    baseline = verify_baseline(
        args.baseline_report,
        config_hash,
        file_sha256(evaluator_path),
        config["dataset"]["blind"]["sha256"],
    )
    _seed_everything(config)

    model_config = config["base_model"]
    training = config["training"]
    snapshot = Path(
        snapshot_download(
            model_config["id"],
            revision=model_config["revision"],
            local_files_only=model_config["local_files_only"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=model_config["trust_remote_code"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    source_train = read_jsonl(resolve(config["dataset"]["train"]["path"]))
    actual_train, _ = seen_schema_holdout(
        source_train,
        config["dataset"]["seen_schema_eval_fraction"],
        config["seeds"]["split"],
    )
    dev = read_jsonl(resolve(config["dataset"]["dev"]["path"]))
    maximum = training["max_sequence_length"]
    prompt_mode = training.get("prompt_mode", "production")
    tokenized_train = [tokenize_training_row(row, tokenizer, maximum, prompt_mode) for row in actual_train]
    tokenized_dev = [tokenize_training_row(row, tokenizer, maximum, prompt_mode) for row in dev]

    quantization = training["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["type"],
        bnb_4bit_use_double_quant=quantization["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=model_config["trust_remote_code"],
        quantization_config=quantization_config,
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
    trainable, total = model.get_nb_trainable_parameters()
    output_dir = resolve(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy=training["eval_strategy"],
        save_strategy=training["save_strategy"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        num_train_epochs=training["epochs"],
        lr_scheduler_type=training["scheduler"],
        warmup_ratio=training["warmup_ratio"],
        logging_steps=training["logging_steps"],
        save_total_limit=training["save_total_limit"],
        load_best_model_at_end=training["load_best_model_at_end"],
        metric_for_best_model=training["metric_for_best_model"],
        greater_is_better=training["greater_is_better"],
        optim=training["optimizer"],
        bf16=training["bf16"],
        fp16=training["fp16"],
        tf32=training["tf32"],
        full_determinism=training["full_determinism"],
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": training["gradient_checkpointing_use_reentrant"]},
        group_by_length=training["group_by_length"],
        dataloader_num_workers=training["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="none",
        seed=config["seeds"]["trainer"],
        data_seed=config["seeds"]["shuffle"],
        include_num_input_tokens_seen=True,
        skip_memory_metrics=False,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TokenizedRows(tokenized_train),
        eval_dataset=TokenizedRows(tokenized_dev),
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    wall_seconds = time.perf_counter() - started
    final_dir = resolve(config["export"]["adapter_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir / "tokenizer")
    trainer.save_state()
    artifact_files = _artifact_files(final_dir)
    artifact_digest = sha256_bytes(canonical(artifact_files).encode())
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "config_sha256": config_hash,
        "preflight_report_sha256": preflight["report_sha256"],
        "protocol_seal_sha256": protocol_seal["report_sha256"],
        "profile_report_sha256": profile["report_sha256"],
        "tiny_overfit_report_sha256": tiny["report_sha256"],
        "baseline_report_sha256": baseline["report_sha256"],
        "base_model": {
            "id": model_config["id"],
            "revision": model_config["revision"],
            "snapshot": str(snapshot),
            "ollama_digest": model_config["ollama_digest"],
        },
        "dataset": {
            "source_train_rows": len(source_train),
            "actual_train_rows": len(tokenized_train),
            "dev_rows": len(tokenized_dev),
            "train_sha256": config["dataset"]["train"]["sha256"],
            "dev_sha256": config["dataset"]["dev"]["sha256"],
        },
        "training": training,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "artifact": {
            "path": str(final_dir),
            "files": artifact_files,
            "aggregate_sha256": artifact_digest,
            "bytes": sum(item["bytes"] for item in artifact_files),
        },
        "determinism": config["environment"]["deterministic_warning"],
        "final_holdout_used": False,
    }
    report["report_sha256"] = sha256_bytes(canonical(report).encode())
    report_path = output_dir / "training-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": report["status"],
                "output": str(report_path),
                "report_sha256": report["report_sha256"],
                "wall_seconds": wall_seconds,
                "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
                "metrics": train_result.metrics,
                "artifact": report["artifact"],
            }
        )
    )


if __name__ == "__main__":
    main()
