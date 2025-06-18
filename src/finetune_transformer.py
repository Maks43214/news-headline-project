# src/finetune_transformer.py

import os
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments

def main():
    
    df = pd.read_csv("data/processed/classification_reduced.csv")
    
    df = df.sample(frac=0.3, random_state=42)
    df = df[df['category_reduced'] != "Other"]

    
    df2 = df[["clean_text", "category_reduced"]].copy()
    df2 = df2.rename(columns={"clean_text": "text", "category_reduced": "label"})

    
    labels = sorted(df2["label"].unique())
    label2id = {label:idx for idx, label in enumerate(labels)}
    id2label = {idx:label for label, idx in label2id.items()}
    df2["label"] = df2["label"].map(label2id)

    
    ds = Dataset.from_pandas(df2)
    ds = ds.train_test_split(test_size=0.2, seed=42)

    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    
    metric_acc = load("accuracy")
    metric_f1  = load("f1")
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    
    training_args = TrainingArguments(
        output_dir="models/distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=500,            
        save_steps=10**9,            
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    
    trainer.train()
    trainer.evaluate()

    
    trainer.save_model("models/distilbert-finetuned")
    tokenizer.save_pretrained("models/distilbert-finetuned")
    print("DistilBERT fine-tuned and saved.")

if __name__ == "__main__":
    main()
