# src/simple_classification.py

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def evaluate(clf, X_train, y_train, X_test, y_test, name: str):
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Train time:    {train_time:.2f}s")
    print(f"Predict time:  {pred_time:.2f}s")
    print(f"Accuracy:      {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return clf

def main():
    
    print("Loading data…")
    
    df = pd.read_csv("data/processed/classification_reduced.csv")
    
    df = df[df['category_reduced']!="Other"]
    y = df['category_reduced']

    
    df = df.sample(frac=0.3, random_state=42)
    print(f"Total examples: {len(df)}")

    X = df['clean_text']
    y = df['category_reduced']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=20000,
        token_pattern=r"(?u)\b\w+\b"
    )
    print("Fitting TF–IDF vectorizer…")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    
    nb   = MultinomialNB()
    knn  = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    dt   = DecisionTreeClassifier(max_depth=50, random_state=42)
    rf   = RandomForestClassifier(n_estimators=100, max_depth=50,
                                  random_state=42, n_jobs=-1)

    
    nb  = evaluate(nb,  X_train_tfidf, y_train, X_test_tfidf, y_test, "MultinomialNB")
    knn = evaluate(knn, X_train_tfidf, y_train, X_test_tfidf, y_test, "kNN (k=5)")
    dt  = evaluate(dt,  X_train_tfidf, y_train, X_test_tfidf, y_test, "DecisionTree")
    rf  = evaluate(rf,  X_train_tfidf, y_train, X_test_tfidf, y_test, "RandomForest")

    
    joblib.dump(nb,  "models/nb.joblib")
    joblib.dump(knn, "models/knn.joblib")
    joblib.dump(dt,  "models/dt.joblib")
    joblib.dump(rf,  "models/rf.joblib")
    print("\nAll models trained and saved under models/")

if __name__ == "__main__":
    main()
