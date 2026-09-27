"""Experimental slot-evidence classifier on a pinned local ruBERT-base encoder.

The cached DeepPavlov checkpoint is split across two Hugging Face snapshots:
config/tokenizer and safetensors weights. This script constructs BertModel from
the local config and loads only encoder tensors through safetensors. It never
deserializes the cached ``pytorch_model.bin`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from scripts import train_slot_evidence_tiny as common

MODEL_ID = "DeepPavlov/rubert-base-cased"
METADATA_REVISION = "4036cab694767a299f2b9e6492909664d9414229"
WEIGHTS_REVISION = "c6047910ee590e0b726e64451a1f6a39d217b257"
CONFIG_SHA256 = "bdf027fe9942ad542d886a6bab649610417ea674a4f3c6209d4d5ffedebaee1d"
WEIGHTS_SHA256 = "51725649c504515fc491cb40bbc4314fcc126c54d51c51c3992c9149a90772b7"
IGNORED_BUFFER = "bert.embeddings.position_ids"


def validate_local_snapshots(metadata: Path, weights: Path) -> dict[str, str]:
    if not metadata.is_dir() or metadata.name != METADATA_REVISION:
        raise ValueError(f"--snapshot must point to pinned config/tokenizer revision {METADATA_REVISION}")
    if not weights.is_dir() or weights.name != WEIGHTS_REVISION:
        raise ValueError(f"--weights-snapshot must point to pinned safetensors revision {WEIGHTS_REVISION}")
    required = {
        "config.json": metadata / "config.json",
        "tokenizer_config.json": metadata / "tokenizer_config.json",
        "special_tokens_map.json": metadata / "special_tokens_map.json",
        "vocab.txt": metadata / "vocab.txt",
        "model.safetensors": weights / "model.safetensors",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete local base snapshots: {missing}")
    hashes = {name: common.sha256_file(path) for name, path in required.items()}
    if hashes["config.json"] != CONFIG_SHA256 or hashes["model.safetensors"] != WEIGHTS_SHA256:
        raise ValueError("pinned ruBERT-base config or safetensors SHA-256 mismatch")
    return hashes


def encoder_state_keys(keys: set[str], expected: set[str]) -> tuple[set[str], set[str]]:
    """Verify that every encoder tensor is present and only known heads are dropped."""
    selected = {key.removeprefix("bert.") for key in keys if key.startswith("bert.") and key != IGNORED_BUFFER}
    if selected != expected:
        raise ValueError(f"encoder tensor mismatch: missing={sorted(expected - selected)}, unexpected={sorted(selected - expected)}")
    ignored = keys - {f"bert.{key}" for key in expected}
    if ignored - {IGNORED_BUFFER} != {key for key in ignored if key.startswith("cls.")}:
        raise ValueError(f"unrecognized checkpoint tensors: {sorted(ignored)}")
    return selected, ignored


def load_safe_encoder(metadata: Path, weights: Path) -> tuple[Any, dict[str, Any]]:
    """Instantiate from JSON, then strictly load safetensors; never read .bin."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(metadata), local_files_only=True, trust_remote_code=False)
    if config.model_type != "bert":
        raise ValueError(f"unexpected model type: {config.model_type}")
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    with safe_open(str(weights / "model.safetensors"), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        expected = set(encoder.state_dict())
        _, ignored = encoder_state_keys(keys, expected)
        state = {key: source.get_tensor(f"bert.{key}") for key in expected}
    encoder.load_state_dict(state, strict=True)
    return encoder, {"encoder_tensors": len(expected), "ignored_tensors": sorted(ignored)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.train.resolve()
    dev_path = args.dev.resolve()
    if train_path == dev_path:
        raise ValueError("train and dev paths must differ")
    train_hash = common.sha256_file(train_path)
    dev_hash = common.sha256_file(dev_path)
    if args.expected_train_sha256 and train_hash != args.expected_train_sha256.lower():
        raise ValueError("train SHA-256 differs from frozen protocol")
    if args.expected_dev_sha256 and dev_hash != args.expected_dev_sha256.lower():
        raise ValueError("dev SHA-256 differs from frozen protocol")
    train = common.read_rows(train_path)
    dev = common.read_rows(dev_path)
    split_report = common.validate_splits(train, dev)
    metadata = args.snapshot.resolve()
    weights = args.weights_snapshot.resolve()
    model_hashes = validate_local_snapshots(metadata, weights)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.max_length < 8 or args.latency_samples < 1:
        raise ValueError("epochs, batch-size, learning-rate, max-length and latency-samples must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import BertTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected Python environment")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(str(metadata), local_files_only=True)
    encoder, loading_report = load_safe_encoder(metadata, weights)

    class PairClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(encoder.config.hidden_size, len(common.LABELS))

        def forward(self, batch: dict[str, Any]) -> Any:
            features = self.encoder(**batch).last_hidden_state[:, 0]
            return self.classifier(self.dropout(features))

    model = PairClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def token_batch(rows: list[dict[str, str]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["premise"] for row in rows],
            [row["hypothesis"] for row in rows],
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {key: tensor.to(device) for key, tensor in encoded.items()}

    def predict(rows: list[dict[str, str]]) -> list[str]:
        model.eval()
        guesses: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                logits = model(token_batch(rows[start : start + args.batch_size]))
                guesses.extend(common.LABELS[index] for index in logits.argmax(dim=-1).tolist())
        return guesses

    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_macro_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch_rows = [train[index] for index in order[start : start + args.batch_size]]
            labels = torch.tensor([common.LABEL_TO_ID[row["label"]] for row in batch_rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_batch(batch_rows))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_rows)
        dev_metrics = common.classification_metrics([row["label"] for row in dev], predict(dev))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "dev": dev_metrics})
        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, output / "best.safetensors")

    training_seconds = time.perf_counter() - started
    model.load_state_dict(load_file(output / "best.safetensors", device=str(device)))
    predictions = predict(dev)
    final_metrics = common.classification_metrics([row["label"] for row in dev], predictions)
    baseline = common.majority_baseline(train, dev)
    sample_rows = dev[: min(len(dev), args.latency_samples)]
    predict(sample_rows[:1])
    timings = []
    model.eval()
    with torch.inference_mode():
        for row in sample_rows:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            item_started = time.perf_counter()
            model(token_batch([row]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - item_started) * 1000)

    report = {
        "status": "trained_experimental_not_default",
        "model": {
            "id": MODEL_ID, "metadata_revision": METADATA_REVISION, "weights_revision": WEIGHTS_REVISION,
            "metadata_snapshot": str(metadata), "weights_snapshot": str(weights),
            "files_sha256": model_hashes, "loading": loading_report, "weight_format": "safetensors_only",
        },
        "input": {
            "train": str(train_path), "train_sha256": train_hash,
            "dev": str(dev_path), "dev_sha256": dev_hash, "split_integrity": split_report,
        },
        "run": {
            "script_sha256": common.sha256_file(Path(__file__)),
            "shared_helpers_sha256": common.sha256_file(Path(common.__file__)),
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "max_length": args.max_length,
            "device": args.device, "best_epoch": best_epoch, "training_seconds": training_seconds,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "majority_baseline": baseline,
        "dev_metrics": final_metrics,
        "dev_metrics_note": "Dev was used to choose the best epoch; these metrics are not an independent final test.",
        "history": history,
        "latency_single_pair_ms": {
            "warmup_excluded": True, "sample_count": len(timings),
            "p50": common.percentile(timings, 0.5), "p95": common.percentile(timings, 0.95),
        },
        "checkpoint_sha256": common.sha256_file(output / "best.safetensors"),
        "predictions": [{"id": row["id"], "gold": row["label"], "predicted": guess} for row, guess in zip(dev, predictions, strict=True)],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path, help="Pinned local config/tokenizer snapshot")
    parser.add_argument("--weights-snapshot", required=True, type=Path, help="Pinned local safetensors snapshot")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-dev-sha256")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--latency-samples", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"base slot evidence experiment failed: {exc}\n")
    print(json.dumps({key: report[key] for key in ("status", "dev_metrics", "majority_baseline", "latency_single_pair_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
