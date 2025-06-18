from datasets import load_dataset
import os

# 1) AG News 
ag = load_dataset("ag_news")
os.makedirs("data/raw/ag_news_hf", exist_ok=True)
ag["train"].to_csv("data/raw/ag_news_hf/train.csv", index=False)
ag["test"].to_csv("data/raw/ag_news_hf/test.csv", index=False)

# 2) CNN/DailyMail for summarization
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
os.makedirs("data/raw/cnn_dailymail_hf", exist_ok=True)
cnn_dailymail["train"].to_csv("data/raw/cnn_dailymail_hf/train.csv", index=False)
cnn_dailymail["validation"].to_csv("data/raw/cnn_dailymail_hf/validation.csv", index=False)
cnn_dailymail["test"].to_csv("data/raw/cnn_dailymail_hf/test.csv", index=False)

print("Saved in data/raw/…_hf/")
