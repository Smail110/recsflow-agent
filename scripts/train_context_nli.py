"""One reproducible LoRA pilot for Russian, role-aware NLI evidence pairs.

The local NLI checkpoint, every tokenizer input and both JSONL inputs must
match pinned SHA-256 values.
This research script does not connect a model to the production workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.train_slot_evidence_tiny import classification_metrics, percentile, sha256_file

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
PINNED_HASHES = {
    "config.json": "4e4430c95100d613df80fa01276f231931e1b535557cf62fbb3cc50323e50cca",
    "model.safetensors": "65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e",
    "tokenizer.json": "3aca3ce69a0a35aeb144a52c4f1d41c4246b8785f8f398315cc8fb6b24057810",
    "tokenizer_config.json": "df9fd3a4482cc4a866788244824573fab8e74dadd1609d4f0e5997f9e96921dc",
    "special_tokens_map.json": "9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9",
    "spm.model": "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3",
}
# Preserve the pretrained NLI head's label semantics. Do not replace its head.
LABEL_TO_ID = {"support": 0, "unknown": 1, "contradiction": 2}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
SEED = 42
EPOCHS = 1
BATCH_SIZE = 4
ACCUMULATION = 4
MAX_LENGTH = 160
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ("query_proj", "key_proj", "value_proj")


def _sealed_path(path: Path) -> bool:
    return any("blind" in part.casefold() or "holdout" in part.casefold() for part in path.parts)


def validate_snapshot(snapshot: Path) -> dict[str, str]:
    if not snapshot.is_dir() or snapshot.name != REVISION:
        raise ValueError(f"snapshot must be the pinned {MODEL_ID} revision {REVISION}")
    missing = [name for name in PINNED_HASHES if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"incomplete local NLI snapshot: {missing}")
    actual = {name: sha256_file(snapshot / name) for name in PINNED_HASHES}
    if actual != PINNED_HASHES:
        raise ValueError("pinned NLI model/tokenizer file SHA-256 mismatch")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "deberta-v2":
        raise ValueError("unexpected pretrained NLI architecture")
    expected = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    if config.get("id2label") != expected:
        raise ValueError("pretrained NLI labels differ from the frozen task mapping")
    return actual


def read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if _sealed_path(path):
        raise ValueError("sealed blind/holdout data are prohibited")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected object on line {number} of {path}")
        for key in ("id", "premise", "hypothesis", "label", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing {key} on line {number} of {path}")
        if row["split"] != split or row["label"] not in LABEL_TO_ID:
            raise ValueError(f"wrong split or label on line {number} of {path}")
        if row["id"] in ids:
            raise ValueError(f"duplicate ID {row['id']}")
        ids.add(row["id"])
        rows.append(row)
    if not rows or set(LABEL_TO_ID) - {row["label"] for row in rows}:
        raise ValueError(f"empty or incomplete three-class split: {path}")
    return rows


def validate_splits(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    def normalized(value: str) -> str:
        return " ".join(value.casefold().replace("ё", "е").split())

    checks = {
        "id": ({row["id"] for row in train}, {row["id"] for row in dev}),
        "premise": ({normalized(row["premise"]) for row in train}, {normalized(row["premise"]) for row in dev}),
        "pair": (
            {(normalized(row["premise"]), normalized(row["hypothesis"])) for row in train},
            {(normalized(row["premise"]), normalized(row["hypothesis"])) for row in dev},
        ),
    }
    for key in ("domain_family_id", "schema_family_id", "template_family_id"):
        if all(isinstance(row.get(key), str) and row[key] for row in train + dev):
            checks[key] = ({row[key] for row in train}, {row[key] for row in dev})
    overlap = {key: len(left & right) for key, (left, right) in checks.items()}
    if any(overlap.values()):
        raise ValueError(f"train/dev split overlap: {overlap}")
    return {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_labels": dict(sorted(Counter(row["label"] for row in train).items())),
        "dev_labels": dict(sorted(Counter(row["label"] for row in dev).items())),
        "overlap": overlap,
    }


def loss_weights(train: list[dict[str, Any]], mode: str) -> dict[str, float]:
    """Return train-only class weights in the pretrained NLI head's label order."""
    if mode not in {"none", "sqrt-inverse"}:
        raise ValueError(f"unknown loss-weighting mode: {mode}")
    counts = Counter(row["label"] for row in train)
    if set(counts) != set(LABEL_TO_ID):
        raise ValueError("train split must contain all NLI labels")
    if mode == "none":
        return dict.fromkeys(LABEL_TO_ID, 1.0)
    raw = {label: 1.0 / math.sqrt(counts[label]) for label in LABEL_TO_ID}
    mean = sum(raw.values()) / len(raw)
    return {label: raw[label] / mean for label in LABEL_TO_ID}


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.train.resolve() == args.dev.resolve():
        raise ValueError("train and dev must be different files")
    if _sealed_path(args.train) or _sealed_path(args.dev):
        raise ValueError("sealed blind/holdout data are prohibited")
    train_hash = sha256_file(args.train)
    dev_hash = sha256_file(args.dev)
    if train_hash != args.expected_train_sha256.lower() or dev_hash != args.expected_dev_sha256.lower():
        raise ValueError("train/dev SHA-256 differs from the frozen protocol")
    train = read_rows(args.train, "train")
    dev = read_rows(args.dev, "dev")
    split_report = validate_splits(train, dev)
    snapshot_hashes = validate_snapshot(args.snapshot)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    return train, dev, {
        "train_sha256": train_hash,
        "dev_sha256": dev_hash,
        "split": split_report,
        "snapshot_sha256": snapshot_hashes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    train, dev, inputs = validate_inputs(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pinned pilot")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    free_before, total_vram = torch.cuda.mem_get_info(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True, trust_remote_code=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        str(args.snapshot), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32,
    ).to(device)
    if base.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise ValueError("loaded NLI classifier has unexpected labels")

    def tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["premise"] for row in rows], [row["hypothesis"] for row in rows],
            truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
        )
        return {key: tensor.to(device) for key, tensor in encoded.items()}

    def predictions(model: Any, rows: list[dict[str, Any]]) -> list[str]:
        model.eval()
        result: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(rows), BATCH_SIZE):
                logits = model(**tokens(rows[start : start + BATCH_SIZE])).logits
                result.extend(ID_TO_LABEL[index] for index in logits.argmax(dim=-1).tolist())
        return result

    # A frozen zero-shot reference uses the same checkpoint, pairs and label map.
    zero_shot = predictions(base, dev)
    adapter = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS, r=LORA_RANK, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=list(LORA_TARGETS),
            modules_to_save=["classifier"], bias="none",
        ),
    )
    trainable = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    if trainable <= 0 or trainable >= sum(parameter.numel() for parameter in adapter.parameters()):
        raise ValueError("LoRA did not freeze the pretrained backbone")
    class_weights = loss_weights(train, args.loss_weighting)
    loss_weight_tensor = (
        torch.tensor([class_weights[ID_TO_LABEL[index]] for index in range(len(ID_TO_LABEL))], device=device)
        if args.loss_weighting == "sqrt-inverse" else None
    )
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = math.ceil(math.ceil(len(train) / BATCH_SIZE) / ACCUMULATION)
    warmup = max(1, math.ceil(steps * WARMUP_RATIO))

    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    batches = [[train[index] for index in order[start : start + BATCH_SIZE]] for start in range(0, len(order), BATCH_SIZE)]
    adapter.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    weighted_loss = 0.0
    for group_start in range(0, len(batches), ACCUMULATION):
        group = batches[group_start : group_start + ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            target = torch.tensor([LABEL_TO_ID[row["label"]] for row in batch], device=device)
            loss = torch.nn.functional.cross_entropy(adapter(**tokens(batch)).logits, target, weight=loss_weight_tensor)
            (loss / len(group)).backward()
            weighted_loss += float(loss.detach()) * len(batch)
        optimizer.step()
        scheduler.step()
    training_seconds = time.perf_counter() - started
    adapter.save_pretrained(str(args.output_dir / "adapter"), safe_serialization=True)
    trained = predictions(adapter, dev)

    # Timings include tokenization and synchronisation; warmup is excluded.
    sample = dev[: min(64, len(dev))]
    predictions(adapter, sample[:1])
    timings: list[float] = []
    for row in sample:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        predictions(adapter, [row])
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000)

    before = classification_metrics([row["label"] for row in dev], zero_shot)
    after = classification_metrics([row["label"] for row in dev], trained)
    report = {
        "status": "experimental_not_default",
        "model": {"id": MODEL_ID, "revision": REVISION, "snapshot": str(args.snapshot), "format": "safetensors_only"},
        "inputs": inputs,
        "training": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "gradient_accumulation": ACCUMULATION,
            "learning_rate": LEARNING_RATE, "max_length": MAX_LENGTH, "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "lora_targets": LORA_TARGETS,
            "loss_weighting": args.loss_weighting, "loss_class_weights": class_weights,
            "optimizer_steps": steps, "trainable_parameters": trainable,
            "train_loss": weighted_loss / len(train), "training_seconds": training_seconds,
        },
        "metrics": {
            "zero_shot": before, "adapted": after,
            "zero_shot_false_support": sum(gold != "support" and pred == "support" for gold, pred in zip((r["label"] for r in dev), zero_shot, strict=True)),
            "adapted_false_support": sum(gold != "support" and pred == "support" for gold, pred in zip((r["label"] for r in dev), trained, strict=True)),
        },
        "dev_predictions": [
            {"id": row["id"], "gold": row["label"], "zero_shot": old, "adapted": new}
            for row, old, new in zip(dev, zero_shot, trained, strict=True)
        ],
        "runtime": {
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "vram_total_bytes": total_vram, "vram_free_before_bytes": free_before,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "latency_single_pair_ms": {"count": len(timings), "p50": percentile(timings, .5), "p95": percentile(timings, .95)},
        },
        "files": {
            "script_sha256": sha256_file(Path(__file__)),
            "adapter_sha256": sha256_file(args.output_dir / "adapter" / "adapter_model.safetensors"),
        },
        "limitations": "Synthetic dev is open and measures generated compositions, not user traffic or independent final quality.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument("--loss-weighting", choices=("none", "sqrt-inverse"), default="none")
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"context NLI pilot failed: {exc}\n")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "runtime": report["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
