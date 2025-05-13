import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re, string, os

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.dropna(subset=['comment_text'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    return text.translate(str.maketrans('', '', string.punctuation)).strip()

def split_and_save(df, labels, test_size=0.2, random_state=42, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    X = df[['id','comment_text']]
    y = df[labels]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df[labels[0]]
    )
    pd.concat([X_train, y_train], axis=1).to_csv(f"{output_dir}/train.csv", index=False)
    pd.concat([X_val,   y_val],   axis=1).to_csv(f"{output_dir}/val.csv",   index=False)

def fit_tfidf(train_texts, max_features=10000, ngram_range=(1,2), stop_words='english'):
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        preprocessor=clean_text
    )
    return tfidf.fit(train_texts)

def save_vectorizer(vectorizer, output_path='data/processed/tfidf.pkl'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(vectorizer, f)
