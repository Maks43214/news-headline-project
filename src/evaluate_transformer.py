# src/evaluate_transformer.py

import numpy as np
import pandas as pd
from transformers import Trainer, DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix

def main():
    
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert-finetuned")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert-finetuned")
    trainer = Trainer(model=model)

    
    df = pd.read_csv("data/processed/classification_reduced.csv")
    df = df.sample(frac=0.3, random_state=42)
    df = df[df['category_reduced'] != "Other"]

    
    labels = sorted(df['category_reduced'].unique())
    label2id = {l:i for i,l in enumerate(labels)}

    
    df2 = df[['clean_text','category_reduced']].copy()
    df2 = df2.rename(columns={'clean_text':'text','category_reduced':'label'})
    df2['label'] = df2['label'].map(label2id)

    
    ds = Dataset.from_pandas(df2, preserve_index=False)
    split = ds.train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]

    
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    test_ds = test_ds.map(preprocess, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = preds_output.label_ids

    
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion matrix:\n")
    print(cm_df)

if __name__ == "__main__":
    main()
