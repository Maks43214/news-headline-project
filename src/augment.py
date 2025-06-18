# src/augment.py

import os
import random
import pandas as pd

def random_delete(text: str, p: float = 0.1) -> str:
    words = text.split()
    remaining = [w for w in words if random.random() > p]
    return " ".join(remaining) if remaining else text

def random_swap(text: str, n_swaps: int = 1) -> str:
    words = text.split()
    for _ in range(n_swaps):
        if len(words) < 2:
            break
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

if __name__ == "__main__":
    
    df = pd.read_csv("data/processed/classification_sample_clean.csv")

    
    df['clean_text'] = df['clean_text'].fillna("").astype(str)
    df = df[df['clean_text'].str.strip() != ""]

    
    df['aug_delete'] = df['clean_text'].apply(lambda t: random_delete(t, p=0.1))
    df['aug_swap']   = df['clean_text'].apply(lambda t: random_swap(t, n_swaps=1))

    
    cols = ['text','headline','category','source']
    
    df_orig = df[cols + ['clean_text']].copy()
    # delete
    df_del  = df[cols].copy()
    df_del['clean_text'] = df['aug_delete']
    # swap
    df_swp  = df[cols].copy()
    df_swp['clean_text'] = df['aug_swap']

    
    augmented = pd.concat([df_orig, df_del, df_swp], ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/classification_augmented.csv"
    augmented.to_csv(out_path, index=False)
    print(f"Сохранено {len(augmented)} записей → {out_path}")
