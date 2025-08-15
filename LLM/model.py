import os, argparse, json
import numpy as np
import pandas as pd
import transformers
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch
print(transformers.__version__)

def autodetect_col(df, preferred, fallbacks):
    if preferred in df.columns: return preferred
    for c in fallbacks:
        if c in df.columns: return c
    raise ValueError(f"Could not find any of these columns in CSV: {[preferred]+fallbacks}\nFound: {list(df.columns)}")

def build_datasets(df, text_col, label_col, test_size=0.1, val_size=0.1, seed=42):
    df = df[[text_col, label_col]].dropna().drop_duplicates()
    df[label_col] = df[label_col].astype(int)
    train_df, temp_df = train_test_split(df, test_size=(test_size+val_size), stratify=df[label_col], random_state=seed)
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=1-rel_val, stratify=temp_df[label_col], random_state=seed)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    return ds

def tokenize_function(examples, tokenizer, text_col, max_length):
    return tokenizer(examples[text_col], truncation=True, padding=False, max_length=max_length)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_and_save(csv_path, text_col, label_col, model_name, output_dir, epochs, batch_size, lr, max_length, seed):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Auto-detect columns if needed
    text_col = text_col or autodetect_col(df, "TEXT", ["text", "Text", "post", "tweet", "content"])
    label_col = label_col or autodetect_col(df, "LABEL", ["label", "labels", "ADE_Label", "ade_label", "target", "class"])

    print(f"Using columns -> TEXT: {text_col} | LABEL: {label_col}")

    ds = build_datasets(df, text_col, label_col, seed=seed)

    id2label = {0: "No_ADE", 1: "ADE"}
    label2id = {"No_ADE": 0, "ADE": 1}

    print(f"Loading model & tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer, text_col, max_length), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("Starting training…")
    trainer.train()

    print("Evaluating on test set…")
    metrics = trainer.evaluate(tokenized["test"])
    print(json.dumps(metrics, indent=2))

    # Detailed test report
    preds = trainer.predict(tokenized["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    print("\nClassification report (test):")
    print(classification_report(y_true, y_pred, target_names=["No_ADE", "ADE"], digits=4))

    print(f"Saving best model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save column names so the inference step knows what was used
    with open(Path(output_dir)/"meta.json", "w", encoding="utf-8") as f:
        json.dump({"text_col": text_col, "label_col": label_col, "id2label": id2label}, f)

def infer_text(text, model_dir):
    print(f"Loading model from: {model_dir}")
    nlp = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=0 if torch.cuda.is_available() else -1, return_all_scores=False)
    out = nlp(text)[0]  # {'label': 'ADE'|'No_ADE', 'score': float}
    label = out["label"]
    score = float(out["score"])
    verdict = "ADE detected" if label == "ADE" else "No ADE"
    print(f"\nText: {text}\nPrediction: {label} (score={score:.4f}) → {verdict}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT/PubMedBERT for ADE detection and run inference.")
    parser.add_argument("--csv", type=str, help="Path to CSV with columns: ID, TEXT, LABEL (0/1).", required=False)
    parser.add_argument("--text-col", type=str, default=None, help="Text column name (default auto-detect).")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name (default auto-detect).")
    parser.add_argument("--model-name", type=str, default="dmis-lab/biobert-base-cased-v1.1",
                        help="HF model name (BioBERT default). Examples: "
                             "'dmis-lab/biobert-base-cased-v1.1', "
                             "'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'")
    parser.add_argument("--output-dir", type=str, default="./ade_model", help="Where to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=str, default=None, help="If set, run inference using saved model (no training).")
    args = parser.parse_args()

    if args.predict:
        # Inference-only mode (expects a trained model in output_dir)
        infer_text(args.predict, args.output_dir)
    else:
        if not args.csv:
            raise SystemExit("Please provide --csv path for training (or use --predict for inference only).")
        train_and_save(
            csv_path=args.csv,
            text_col=args.text_col,
            label_col=args.label_col,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            seed=args.seed,
        )
        # Quick interactive demo
        print("\nType a sentence to classify (or just press Enter to quit):")
        while True:
            s = input("> ").strip()
            if not s: break
            infer_text(s, args.output_dir)