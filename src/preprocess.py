import re
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import download as nltk_download

# 1) Load NLTK + spaCy resources
nltk_download('stopwords')
STOPWORDS = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_html(text: str) -> str:
    """Remove HTML tags and boilerplate using BeautifulSoup."""
    return BeautifulSoup(text, "lxml").get_text(separator=" ")

def clean_special_chars(text: str) -> str:
    """Remove special characters, extra spaces, and convert to lowercase."""
    text = re.sub(r"\s+", " ", text)             # multiple spaces → one
    text = re.sub(r"[^A-Za-z0-9.,;:()'\"%\- ]+", " ", text)
    return text.lower().strip()

def lemmatize_and_remove_stopwords(text: str) -> str:
    """Lemmatize and remove stopwords."""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2]
    return " ".join(lemmas)

def preprocess_series(series: pd.Series) -> pd.Series:
    """Applies the full cleaning pipeline to a Pandas series."""
    return (series
            .astype(str)
            .apply(clean_html)
            .apply(clean_special_chars)
            .apply(lemmatize_and_remove_stopwords)
           )

if __name__ == "__main__":
    # Load classification data
    df = pd.read_csv("data/processed/classification_news.csv")

    # Sample a small portion for testing
    df_sample = df

    # Preprocess the 'text' column
    print("Cleaning sample...")
    df_sample['clean_text'] = preprocess_series(df_sample['text'])

    # Save the results
    df_sample.to_csv("data/processed/classification_sample_clean.csv", index=False)
    print("Sample cleaned and saved to classification_sample_clean.csv")
