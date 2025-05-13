import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_data(csv_path, sample_size=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['comment_text'])
    df[LABELS] = df[LABELS].fillna(0).astype(int)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df

def preprocess(df):
    df['labels'] = df[LABELS].values.tolist()
    return df[['comment_text', 'labels']]

def tokenize_data(df, tokenizer):
    dataset = Dataset.from_pandas(df)

    def tokenize(example):
        return tokenizer(example["comment_text"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(tokenize, batched=True)

    def cast_labels(example):
        example["labels"] = [float(label) for label in example["labels"]]
        return example

    tokenized_dataset = tokenized_dataset.map(cast_labels)
    return tokenized_dataset

def prepare_dataset(path, tokenizer, sample_size=None):
    df = load_data(path, sample_size)
    df = preprocess(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_dataset = tokenize_data(train_df, tokenizer)
    val_dataset = tokenize_data(val_df, tokenizer)
    test_dataset = tokenize_data(test_df, tokenizer)

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
