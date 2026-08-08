"""
src/train.py - MAI203 Amazon Fine Food Reviews
Training stage: trains LogReg + TF-IDF, logs to MLflow, saves best checkpoint.
Called by: dvc repro train
"""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load params from params.yaml only
PARAMS = yaml.safe_load(open('params.yaml'))
DATA   = PARAMS['data']
TFIDF  = PARAMS['tfidf']
LOGREG = PARAMS['logreg']
NB     = PARAMS['naive_bayes']
ML     = PARAMS['mlflow']


def load_data():
    csv_path = Path(DATA['cleaned_csv'])
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run: dvc repro prepare")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['sentiment', 'processed_text'])
    return df


def split_data(df: pd.DataFrame):
    X = df['processed_text'].fillna('')
    y = df['sentiment']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=DATA['test_size'] + DATA['val_size'],
        random_state=DATA['random_state'],
        stratify=y
    )
    val_ratio = DATA['val_size'] / (DATA['test_size'] + DATA['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        random_state=DATA['random_state'],
        stratify=y_temp
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logreg(X_train, y_train, X_val, y_val):
    """Train TF-IDF + Logistic Regression with grid search over C."""
    tfidf_vec = TfidfVectorizer(
        max_features=TFIDF['max_features'],
        ngram_range=(TFIDF['ngram_min'], TFIDF['ngram_max']),
        min_df=TFIDF['min_df'],
        sublinear_tf=TFIDF['sublinear_tf']
    )
    X_tr_vec  = tfidf_vec.fit_transform(X_train)
    X_val_vec = tfidf_vec.transform(X_val)

    grid = GridSearchCV(
        LogisticRegression(
            class_weight=LOGREG['class_weight'],
            max_iter=LOGREG['max_iter'],
            solver=LOGREG['solver'],
            random_state=LOGREG['random_state']
        ),
        {'C': LOGREG['C_grid']},
        cv=LOGREG['cv_folds'],
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X_tr_vec, y_train)
    best_c = grid.best_params_['C']
    print(f"  Best C: {best_c}")

    y_pred = grid.best_estimator_.predict(X_val_vec)
    val_f1 = f1_score(y_val, y_pred, average='macro')
    val_acc = accuracy_score(y_val, y_pred)
    return grid.best_estimator_, tfidf_vec, val_f1, val_acc, best_c


def train_naive_bayes(X_train, y_train, X_val, y_val):
    """Train BoW + Multinomial Naive Bayes - baseline model."""
    bow_vec   = CountVectorizer(max_features=PARAMS['bow']['max_features'],
                                 min_df=PARAMS['bow']['min_df'])
    X_tr_bow  = bow_vec.fit_transform(X_train)
    X_val_bow = bow_vec.transform(X_val)

    nb_model = MultinomialNB(alpha=NB['alpha'])
    nb_model.fit(X_tr_bow, y_train)

    y_pred  = nb_model.predict(X_val_bow)
    val_f1  = f1_score(y_val, y_pred, average='macro')
    val_acc = accuracy_score(y_val, y_pred)
    return nb_model, bow_vec, val_f1, val_acc


def train():
    Path('results/models').mkdir(parents=True, exist_ok=True)
    Path('results/metrics').mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(ML['tracking_uri'])
    mlflow.set_experiment(ML['experiment_name'])

    print("Loading data...")
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Train Naive Bayes baseline
    print("\nTraining Naive Bayes baseline...")
    with mlflow.start_run(run_name='naive_bayes_baseline'):
        nb_model, bow_vec, nb_f1, nb_acc = train_naive_bayes(X_train, y_train, X_val, y_val)
        mlflow.log_params({'model': 'naive_bayes', 'alpha': NB['alpha'],
                           'max_features': PARAMS['bow']['max_features']})
        mlflow.log_metrics({'val_macro_f1': nb_f1, 'val_accuracy': nb_acc})
        print(f"  NB Val Macro-F1: {nb_f1:.4f}  Accuracy: {nb_acc:.4f}")

    # Train LogReg + TF-IDF (primary model)
    print("\nTraining LogReg + TF-IDF (primary model)...")
    with mlflow.start_run(run_name='logreg_tfidf'):
        lr_model, tfidf_vec, lr_f1, lr_acc, best_c = train_logreg(
            X_train, y_train, X_val, y_val)
        mlflow.log_params({'model': 'logreg_tfidf', 'C': best_c,
                           'max_features': TFIDF['max_features'],
                           'ngram_range': f"{TFIDF['ngram_min']},{TFIDF['ngram_max']}"})
        mlflow.log_metrics({'val_macro_f1': lr_f1, 'val_accuracy': lr_acc})
        mlflow.sklearn.log_model(lr_model, 'model')
        print(f"  LR Val Macro-F1: {lr_f1:.4f}  Accuracy: {lr_acc:.4f}")

    # Save best model (LogReg consistently wins)
    checkpoint = {
        'model':       lr_model,
        'vectorizer':  tfidf_vec,
        'model_name':  'LogReg + TF-IDF',
        'val_f1':      lr_f1,
        'val_acc':     lr_acc,
        'best_c':      best_c,
    }
    model_path = Path(PARAMS['api']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f_out:
        pickle.dump(checkpoint, f_out)
    print(f"\nBest model saved: {model_path}")

    # Save training history
    history = {'logreg': {'val_f1': lr_f1, 'val_acc': lr_acc, 'best_c': best_c},
               'naive_bayes': {'val_f1': nb_f1, 'val_acc': nb_acc}}
    with open('results/models/history.json', 'w') as f_out:
        json.dump(history, f_out, indent=2)

    # Save train metrics for DVC
    train_metrics = {'val_macro_f1': round(lr_f1, 4), 'val_accuracy': round(lr_acc, 4)}
    with open('results/metrics/train_metrics.json', 'w') as f_out:
        json.dump(train_metrics, f_out, indent=2)

    print("\nTraining complete.")
    print(f"  Best model: {checkpoint['model_name']}")
    print(f"  Val Macro-F1: {lr_f1:.4f}")
    print("  View MLflow UI: mlflow ui --port 5001  (macOS) / 8080  (Windows)")


if __name__ == '__main__':
    train()
