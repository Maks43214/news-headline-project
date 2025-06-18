# src/reduce_categories.py

import pandas as pd


df = pd.read_csv("data/processed/classification_augmented.csv")


df['cat_norm'] = (
    df['category']
      .astype(str)
      .str.lower()
      .str.replace(r'[^a-z\s]', ' ', regex=True)
      .str.replace(r'\s+', ' ', regex=True)
      .str.strip()
)


mapping = {
    # Business
    **dict.fromkeys(["business", "money"], "Business"),
    # Politics
    "politics": "Politics",
    # Sports
    "sports": "Sports",
    # Tech & Science
    **dict.fromkeys(["sci tech", "science", "tech"], "Tech & Science"),
    # Entertainment
    **dict.fromkeys([
        "entertainment","comedy","culture arts","arts culture",
        "arts","media","style beauty","style","fifty"
    ], "Entertainment"),
    # Health & Wellness
    **dict.fromkeys(["healthy living","wellness","good news"], "Health & Wellness"),
    # Food & Drink
    **dict.fromkeys(["food drink","taste"], "Food & Drink"),
    # Lifestyle & Culture
    **dict.fromkeys([
        "home living","travel","weddings","green","environment",
        "education","divorce"
    ], "Lifestyle & Culture"),
    # World
    **dict.fromkeys(["world news","world","the worldpost","worldpost"], "World"),
    # Parenting & Family
    **dict.fromkeys(["parenting","parents"], "Parenting & Family"),
    # всё остальное
}


df['category_reduced'] = df['cat_norm'].map(mapping).fillna("Other")



print("Nowe rozkłady kategorii:")
print(
    df.loc[df['category_reduced'] == "Other", 'cat_norm']
      .value_counts()
      .head(10)
)

out = df[['text','headline','clean_text','category_reduced','source']]
out.to_csv("data/processed/classification_reduced.csv", index=False)
print("Сохранено → data/processed/classification_reduced.csv")

print("\nNew category counts:")
print(out['category_reduced'].value_counts())
