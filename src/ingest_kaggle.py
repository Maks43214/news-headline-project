import os
import glob
import pandas as pd

def load_ag_news_hf():

    dfs = []
    for split in ["train", "test"]:
        path = f"data/raw/ag_news_hf/{split}.csv"
        tmp = pd.read_csv(path)
        
        if "text" in tmp.columns and "label" in tmp.columns:
            df = pd.DataFrame({
                "text":     tmp["text"],
                "headline": tmp["text"] .str.split(". ").str[0],  # HF AG News не содержит заголовков, можно взять первое предложение
                "category": tmp["label"].map({0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}),
                "source":   "AG_News"
            })
        else:
            
            df = pd.DataFrame({
                "text":       tmp["description"],
                "headline":   tmp["title"],
                "category":   tmp["class"].map({1:"World",2:"Sports",3:"Business",4:"Sci/Tech"}),
                "source":     "AG_News"
            })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_all_the_news():
    # All The News: News_Category_Dataset_v2.json
    path = "data/raw/all_news/News_Category_Dataset_v3.json"
    df = pd.read_json(path, lines=True)
    return pd.DataFrame({
        "text":       df["short_description"],
        "headline":   df["headline"],
        "category":   df["category"],
        "source":     "All_The_News"
    })

def load_cnn_dailymail():
    
    dfs = []
    for split in ["train", "validation", "test"]:
        path = f"data/raw/cnn_dailymail_hf/{split}.csv"
        tmp = pd.read_csv(path)
        dfs.append(pd.DataFrame({
            "text":       tmp["article"],
            "headline":   tmp["highlights"],
            "category":   None,
            "source":     f"CNN_DailyMail_{split}"
        }))
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    ag   = load_ag_news_hf()
    alln = load_all_the_news()
    cnn  = load_cnn_dailymail()

    
    classification_df = pd.concat([ag, alln], ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    classification_df.to_csv("data/processed/classification_news.csv", index=False)
    print(f"{len(classification_df)} записей для классификации → data/processed/classification_news.csv")

    cnn.to_csv("data/processed/summarization_news.csv", index=False)
    print(f"{len(cnn)} записей для summarization → data/processed/summarization_news.csv")
