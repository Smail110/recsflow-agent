"""Fine-tune an intent classifier with Hugging Face Trainer + PEFT LoRA.

Validation is deliberately independent of the optional HF stack. Synthetic data
only checks the pipeline; it is not a claim about real-world model quality.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.prepare_hf_dataset import validate_rows
except ModuleNotFoundError:  # ``python scripts/train_lora.py`` from the repo root
    from prepare_hf_dataset import validate_rows


def load_and_validate(path: str | Path) -> dict:
    source = Path(path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return validate_rows(rows, source=source)


def validation_report(validated: dict, *, model: str, output: str) -> dict:
    rows = validated["rows"]
    train_rows = [rows[i] for i in validated["train_indices"]]
    validation_rows = [rows[i] for i in validated["validation_indices"]]
    return {
        "status": "validated",
        "rows": len(rows),
        "labels": validated["labels"],
        "model": model,
        "output": output,
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
        },
        "label_counts": {
            "train": dict(sorted(Counter(row["label"] for row in train_rows).items())),
            "validation": dict(sorted(Counter(row["label"] for row in validation_rows).items())),
        },
        "group_counts": {
            "train": len(validated["train_groups"]),
            "validation": len(validated["validation_groups"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dialogues.jsonl")
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--output", default="artifacts/intent-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        validated = load_and_validate(args.data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc
    report = validation_report(validated, model=args.model, output=args.output)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Установите requirements-hf.txt для обучения Hugging Face") from exc

    rows = validated["rows"]
    labels = validated["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    # These are the indices already checked above. Keeping this construction
    # explicit prevents a later library call from silently making a row split.
    train_rows = [rows[i] for i in validated["train_indices"]]
    validation_rows = [rows[i] for i in validated["validation_indices"]]
    train_dataset = Dataset.from_list([{"text": row["text"], "label": label_to_id[row["label"]]} for row in train_rows])
    validation_dataset = Dataset.from_list([{"text": row["text"], "label": label_to_id[row["label"]]} for row in validation_rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={v: k for k, v in label_to_id.items()},
        label2id=label_to_id,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        ),
    )
    model.print_trainable_parameters()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "labels.json").write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "status": "trained", "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
