"""
src/prepare.py - MAI203 Amazon Fine Food Reviews
Data preparation stage: load, clean, tokenize, lemmatize, and save cleaned CSV.
Called by: dvc repro prepare
"""

import re
import sys
from pathlib import Path

import nltk
import pandas as pd
import yaml
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load params from params.yaml only - never hardcode
with open('params.yaml') as f:
    PARAMS = yaml.safe_load(f)
DATA   = PARAMS['data']
PRE    = PARAMS['preprocessing']

STOP_WORDS = set(stopwords.words(PRE['stopwords_lang']))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(tag: str):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str) -> str:
    text = str(text)[:PRE['max_text_length']].lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_lemmatize(text: str) -> str:
    tokens = word_tokenize(clean_text(text))
    tokens = [t for t in tokens
              if t not in STOP_WORDS and len(t) >= PRE['min_token_length']]
    tagged = pos_tag(tokens)
    lemmas = [LEMMATIZER.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return ' '.join(lemmas)


def map_sentiment(score: int):
    if score >= DATA['score_positive_min']: return 'positive'
    if score <= DATA['score_negative_max']: return 'negative'
    return None


def map_3class(score: int) -> str:
    if score >= DATA['score_positive_min']: return 'positive'
    if score <= DATA['score_negative_max']: return 'negative'
    return 'neutral'


def prepare():
    raw_path = Path(DATA['raw_csv'])
    out_path = Path(DATA['cleaned_csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found.")
        print("Download from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews")
        sys.exit(1)

    print(f"Loading {raw_path} ...")
    df = pd.read_csv(raw_path)
    print(f"  Full dataset: {len(df):,} rows")

    df = df.sample(DATA['sample_size'], random_state=DATA['random_state']).reset_index(drop=True)
    print(f"  Sample size:  {len(df):,} rows")

    df['cleaned_text']   = df['Text'].apply(clean_text)
    print("  Text cleaned.")

    print("  Tokenizing and lemmatizing (this takes a few minutes)...")
    df['processed_text'] = df['Text'].apply(tokenize_lemmatize)

    df['sentiment']   = df['Score'].apply(map_sentiment)
    df['label_3class'] = df['Score'].apply(map_3class)

    df = df[['Id', 'ProductId', 'Score', 'Summary', 'Text',
             'cleaned_text', 'processed_text', 'sentiment', 'label_3class']]

    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df):,} rows)")
    print(f"  Sentiment distribution:\n{df['sentiment'].value_counts()}")


if __name__ == '__main__':
    prepare()
