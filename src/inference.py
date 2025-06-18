from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import pandas as pd


model = DistilBertForSequenceClassification.from_pretrained("models/distilbert-finetuned").eval().to("cuda")
tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert-finetuned")


samples = [
    "The stock market saw a 5% increase in major indexes after positive earnings reports.",
    "The local team secured a last-minute victory in the championship game.",
    "Scientists have discovered a new species of bacteria in deep-sea vents.",
    "Parents are concerned about the new school curriculum changes this semester.",
    "The latest superhero movie has broken box office records worldwide.",
    "A new cooking show featuring international cuisines has become a ratings hit."
]


enc = tokenizer(samples, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**enc).logits
preds = torch.argmax(logits, dim=-1).cpu().tolist()


labels = model.config.id2label
df = pd.DataFrame({
    "Sample Text": samples,
    "Predicted Category": [labels[str(p)] if str(p) in labels else labels[p] for p in preds]
})
print(df.to_string(index=False))
