"""Fine-tune an intent classifier with Hugging Face Trainer + PEFT LoRA.

This script is intentionally separate from the demo runtime. Use real, de-identified,
consented labelled conversations for a meaningful model; synthetic data only checks the pipeline.
"""
import argparse, json, os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dialogues.jsonl")
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--output", default="artifacts/intent-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = sorted({row["label"] for row in rows})
    if len(rows) < len(labels) * 5 or not labels:
        raise SystemExit("Need at least 5 labelled rows per class")
    if args.dry_run:
        print(json.dumps({"status":"validated", "rows":len(rows), "labels":labels, "model":args.model, "output":args.output}, ensure_ascii=False, indent=2))
        return
    try:
        from datasets import ClassLabel, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("Установите requirements-hf.txt для обучения Hugging Face") from exc
    label_to_id = {label: index for index, label in enumerate(labels)}
    dataset = Dataset.from_list([{"text": row["text"], "label": label_to_id[row["label"]]} for row in rows])
    dataset = dataset.cast_column("label", ClassLabel(names=labels)).train_test_split(test_size=.2, seed=42, stratify_by_column="label")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels), id2label={v:k for k,v in label_to_id.items()}, label2id=label_to_id)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=.1, target_modules=["query", "value"], modules_to_save=["classifier"]))
    model.print_trainable_parameters()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs, learning_rate=2e-4, per_device_train_batch_size=8, per_device_eval_batch_size=16, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to="none", seed=42)
    trainer = Trainer(model=model, args=training, train_dataset=tokenized["train"], eval_dataset=tokenized["test"], tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train(); metrics = trainer.evaluate(); trainer.save_model(str(output)); tokenizer.save_pretrained(str(output))
    (output / "labels.json").write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"trained", "rows":len(rows), "labels":labels, "metrics":metrics, "output":str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
