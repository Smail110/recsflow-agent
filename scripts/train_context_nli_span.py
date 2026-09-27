"""Final research-only citation pooling experiment; the application is unchanged.

Reuse the pinned paired-citation training protocol, adding one zero-initialized
residual to the pretrained CLS representation. No dataset or labels are changed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts import train_context_nli_pair as baseline
from scripts.eval_massive_citation_nli import paired_metrics, read_pairs
from scripts.train_slot_evidence_tiny import classification_metrics, percentile


def citation_bounds(premise: str) -> tuple[int, int]:
    if premise.count("<evidence>") != 1 or premise.count("</evidence>") != 1:
        raise ValueError("exactly one marked citation is required")
    start = premise.index("<evidence>") + len("<evidence>")
    end = premise.index("</evidence>")
    if start >= end:
        raise ValueError("citation must be nonempty")
    return start, end


def span_masks(
    premise: str, hypothesis: str, offsets: list[tuple[int, int]], sequences: list[int | None],
) -> tuple[list[bool], list[bool]]:
    """Select tokens strictly within the citation and the hypothesis sequence.

    Letters and digits must all be covered. SentencePiece may merge boundary
    punctuation with a marker (for example ``.</``); that token is not pooled.
    """
    if len(offsets) != len(sequences):
        raise ValueError("offset and sequence lengths differ")
    start, end = citation_bounds(premise)
    citation = [seq == 0 and start <= left < right <= end for (left, right), seq in zip(offsets, sequences, strict=True)]
    claim = [seq == 1 and left < right for (left, right), seq in zip(offsets, sequences, strict=True)]
    for text, lower, upper, mask in ((premise, start, end, citation), (hypothesis, 0, len(hypothesis), claim)):
        covered = set()
        for (left, right), selected in zip(offsets, mask, strict=True):
            if selected:
                covered.update(range(left, right))
        if not any(mask) or any(text[pos].isalnum() and pos not in covered for pos in range(lower, upper)):
            raise ValueError("citation or hypothesis is truncated or crosses a marker boundary")
    return citation, claim


def encode(tokenizer: Any, rows: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    import torch

    encoded = tokenizer(
        [row["premise"] for row in rows], [row["hypothesis"] for row in rows],
        truncation=True, padding=True, max_length=baseline.MAX_LENGTH,
        return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").tolist()
    masks = []
    for index, (row, positions) in enumerate(zip(rows, offsets, strict=True)):
        try:
            masks.append(span_masks(row["premise"], row["hypothesis"], positions, encoded.sequence_ids(index)))
        except ValueError as exc:
            raise ValueError(f"mask validation failed for {row['id']}: {exc}") from exc
    result = {key: value.to(device) for key, value in encoded.items()}
    result["citation_mask"] = torch.tensor([mask[0] for mask in masks], device=device)
    result["hypothesis_mask"] = torch.tensor([mask[1] for mask in masks], device=device)
    return result


def make_model(adapter: Any) -> Any:
    import torch

    class SpanResidualModel(torch.nn.Module):
        def __init__(self, underlying: Any):
            super().__init__()
            self.adapter = underlying
            size = underlying.get_base_model().config.hidden_size
            # Initialization must not alter the baseline training RNG stream.
            with torch.random.fork_rng():
                self.span_projection = torch.nn.Linear(4 * size, size, bias=False)
                torch.nn.init.zeros_(self.span_projection.weight)

        def forward(self, citation_mask: Any, hypothesis_mask: Any, **inputs: Any) -> Any:
            core = self.adapter.get_base_model()
            hidden = core.deberta(**inputs, return_dict=True).last_hidden_state

            def mean(mask: Any) -> Any:
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                if bool((weights.sum(dim=1) == 0).any()):
                    raise ValueError("empty pooling mask")
                return (hidden * weights).sum(dim=1) / weights.sum(dim=1)

            citation, claim = mean(citation_mask), mean(hypothesis_mask)
            features = torch.cat((citation, claim, citation - claim, citation * claim), dim=-1)
            selected = hidden[:, 0] + self.span_projection(features)
            pooled = core.pooler(selected.unsqueeze(1))
            return SimpleNamespace(logits=core.classifier(core.dropout(pooled)))

    return SpanResidualModel(adapter)


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    train, dev, inputs = baseline.validate_inputs(args)
    external = read_pairs(args.external, args.expected_external_sha256)
    old_dev = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    old_external = json.loads(args.baseline_external.read_text(encoding="utf-8"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    torch.cuda.manual_seed_all(baseline.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    free_before, total_vram = torch.cuda.mem_get_info(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True, trust_remote_code=False)

    def fresh_base() -> Any:
        return AutoModelForSequenceClassification.from_pretrained(
            str(args.snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
        ).to(device)

    base = fresh_base()
    adapter = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=baseline.LORA_RANK, lora_alpha=baseline.LORA_ALPHA,
        lora_dropout=baseline.LORA_DROPOUT, target_modules=list(baseline.LORA_TARGETS),
        modules_to_save=["classifier"], bias="none",
    ))
    model = make_model(adapter).to(device)
    model.eval()
    # Mask validation for every row before any training/inference.
    for cohort in (train, dev, external):
        for start in range(0, len(cohort), 32):
            encode(tokenizer, cohort[start:start + 32], device)
    probe = encode(tokenizer, dev[:4], device)
    with torch.inference_mode():
        reference = adapter(**{key: value for key, value in probe.items() if not key.endswith("_mask") or key == "attention_mask"}).logits
        initial = model(**probe).logits
    initial_delta = float((initial - reference).abs().max())
    if initial_delta != 0.0:
        raise ValueError(f"zero-init logits differ: {initial_delta}")
    print("Validated all masks and exact zero-init logits parity", flush=True)

    def infer(current: Any, rows: list[dict[str, Any]], batch_size: int = 4) -> tuple[list[str], list[list[float]]]:
        current.eval()
        labels, logits = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                scores = current(**encode(tokenizer, rows[start:start + batch_size], device)).logits
                labels.extend(baseline.ID_TO_LABEL[index] for index in scores.argmax(-1).tolist())
                logits.extend(scores.cpu().tolist())
        return labels, logits

    pairs = baseline.citation_pairs(train)
    random.Random(baseline.SEED).shuffle(pairs)
    batches = [[row for pair in pairs[start:start + 2] for row in pair] for start in range(0, len(pairs), 2)]
    steps = math.ceil(len(batches) / baseline.ACCUMULATION)
    warmup = max(1, math.ceil(steps * baseline.WARMUP_RATIO))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=baseline.LEARNING_RATE, weight_decay=baseline.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started, loss_sum = time.perf_counter(), 0.0
    model.train()
    for group_start in range(0, len(batches), baseline.ACCUMULATION):
        group = batches[group_start:group_start + baseline.ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            target = torch.tensor([baseline.LABEL_TO_ID[row["label"]] for row in batch], device=device)
            scores = model(**encode(tokenizer, batch, device)).logits
            loss = torch.nn.functional.cross_entropy(scores, target)
            actionable = [index for index in range(0, len(batch), 2) if batch[index]["label"] != "unknown"]
            if actionable:
                positive = torch.tensor(actionable, device=device)
                negative = positive + 1
                label_ids = target[positive]
                delta = scores[positive, label_ids] - scores[positive, 1] - scores[negative, label_ids] + scores[negative, 1]
                loss = loss + baseline.PAIR_WEIGHT * torch.nn.functional.softplus(baseline.PAIR_MARGIN - delta).mean()
            (loss / len(group)).backward()
            loss_sum += float(loss.detach()) * len(batch)
        optimizer.step()
        scheduler.step()
        step = group_start // baseline.ACCUMULATION + 1
        if step % 100 == 0:
            print(f"Training {step}/{steps}", flush=True)
    elapsed = time.perf_counter() - started
    adapter.save_pretrained(str(args.output_dir / "adapter"), safe_serialization=True)
    save_file(model.span_projection.state_dict(), str(args.output_dir / "span_projection.safetensors"))
    trained, logits = infer(model, dev)
    external_labels, external_logits = infer(model, external, 16)
    _, saved_probe = infer(model, dev[:4])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_vram = torch.cuda.max_memory_allocated(device)
    del model, adapter, base, optimizer, scheduler
    torch.cuda.empty_cache()
    reloaded_adapter = PeftModel.from_pretrained(fresh_base(), str(args.output_dir / "adapter"), is_trainable=False)
    reloaded = make_model(reloaded_adapter).to(device)
    reloaded.span_projection.load_state_dict(load_file(str(args.output_dir / "span_projection.safetensors"), device="cuda:0"))
    _, loaded_probe = infer(reloaded, dev[:4])
    reload_delta = float((torch.tensor(saved_probe) - torch.tensor(loaded_probe)).abs().max())
    if reload_delta != 0.0:
        raise ValueError(f"reload logits differ: {reload_delta}")
    infer(reloaded, dev[:1])
    timings = []
    for row in dev[:64]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer(reloaded, [row])
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
    old_by_id = {row["id"]: row["adapted"] for row in old_dev["dev_predictions"]}
    external_by_id = {row["id"]: row["paired_margin"] for row in old_external["predictions"]}
    old_labels = [old_by_id[row["id"]] for row in dev]
    old_external_labels = [external_by_id[row["id"]] for row in external]
    gold = [row["label"] for row in dev]
    report = {
        "status": "exploratory_research_only",
        "inputs": inputs,
        "training": {"seed": 42, "epochs": 1, "optimizer_steps": steps, "trainable_parameters": trainable,
                     "seconds": elapsed, "loss": loss_sum / len(train)},
        "parity": {"zero_init_max_abs_logit_delta": initial_delta, "reload_max_abs_logit_delta": reload_delta},
        "metrics": {"baseline": classification_metrics(gold, old_labels), "adapted": classification_metrics(gold, trained)},
        "dev_predictions": [{"id": row["id"], "gold": row["label"], "baseline": old, "adapted": new, "logits": scores}
                            for row, old, new, scores in zip(dev, old_labels, trained, logits, strict=True)],
        "external": {"baseline": paired_metrics(external, old_external_labels), "adapted": paired_metrics(external, external_labels)},
        "external_predictions": [{"id": row["id"], "source_id": row["source_id"], "group_id": row["group_id"],
                                  "gold": row["label"], "baseline": old, "adapted": new, "logits": scores}
                                 for row, old, new, scores in zip(external, old_external_labels, external_labels, external_logits, strict=True)],
        "runtime": {"gpu": torch.cuda.get_device_name(), "vram_total_bytes": total_vram, "vram_free_before_bytes": free_before,
                    "peak_cuda_allocated_bytes": peak_vram, "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
                    "latency_single_pair_ms": {"count": 64, "p50": percentile(timings, .5), "p95": percentile(timings, .95)}},
    }
    dump(args.output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "dev", "snapshot", "output-dir", "external", "baseline-report", "baseline-external"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("train", "dev", "external"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("training", "parity", "metrics", "external", "runtime")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
