"""
src/evaluate.py - MAI203 Amazon Fine Food Reviews
Evaluation stage: classifier metrics, BM25 IR metrics, confusion matrix, PR curves.
Called by: dvc repro evaluate
"""

import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from rank_bm25 import BM25Okapi

PARAMS = yaml.safe_load(open('params.yaml'))
DATA   = PARAMS['data']
BM25P  = PARAMS['bm25']


def load_test_data():
    csv_path = Path(DATA['cleaned_csv'])
    df = pd.read_csv(csv_path).dropna(subset=['sentiment', 'processed_text'])
    _, temp = train_test_split(df, test_size=DATA['test_size'] + DATA['val_size'],
                                random_state=DATA['random_state'], stratify=df['sentiment'])
    val_ratio = DATA['val_size'] / (DATA['test_size'] + DATA['val_size'])
    _, test_df = train_test_split(temp, test_size=1 - val_ratio,
                                   random_state=DATA['random_state'],
                                   stratify=temp['sentiment'])
    return test_df.reset_index(drop=True), df


# ── IR metric helpers ───────────────────────────────────────────────────────

def precision_at_k(rel, k):  return sum(rel[:k]) / k
def recall_at_k(rel, k):
    total = sum(rel); return sum(rel[:k]) / total if total else 0.0
def average_precision(rel):
    hits, precs = 0, []
    for i, r in enumerate(rel, 1):
        if r: hits += 1; precs.append(hits / i)
    return float(np.mean(precs)) if precs else 0.0
def mrr(rel):
    for i, r in enumerate(rel, 1):
        if r: return 1.0 / i
    return 0.0
def ndcg_at_k(rel, k):
    dcg  = sum(r / math.log2(i+1) for i, r in enumerate(rel[:k], 1))
    idcg = sum(r / math.log2(i+1) for i, r in enumerate(sorted(rel, reverse=True)[:k], 1))
    return dcg / idcg if idcg else 0.0


# ── Classifier evaluation ───────────────────────────────────────────────────

def evaluate_classifier(model, vectorizer, test_texts, test_labels, plots_dir):
    X_test = vectorizer.transform(test_texts)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(test_labels, y_pred)
    mf1    = f1_score(test_labels, y_pred, average='macro')
    prec   = precision_score(test_labels, y_pred, average='macro', zero_division=0)
    rec    = recall_score(test_labels, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*52}")
    print("  CLASSIFIER EVALUATION")
    print(f"{'='*52}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro-F1   : {mf1:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print("\n" + classification_report(test_labels, y_pred, zero_division=0))

# Confusion matrix
    labels_order = sorted(set(test_labels))
    cm = confusion_matrix(test_labels, y_pred, labels=labels_order)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    # PR curves
    y_score = model.predict_proba(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#f87171', '#34d399']
    for i, (lbl, color) in enumerate(zip(labels_order, colors)):
        prec_c, rec_c, _ = precision_recall_curve(
            [1 if t == lbl else 0 for t in test_labels],
            y_score[:, i]
        )
        axes[i].plot(rec_c, prec_c, color=color, lw=2)
        axes[i].set_title(f'PR Curve: {lbl}')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
    plt.suptitle('Per-Class Precision-Recall Curves')
    plt.tight_layout()
    plt.savefig(plots_dir / 'pr_curves.png', dpi=150)
    plt.close()

    return {'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4),
            'precision': round(prec, 4), 'recall': round(rec, 4)}

# ── BM25 evaluation ─────────────────────────────────────────────────────────

EVAL_QUERIES = {
    "great taste delicious flavor":  [1,1,1,1,1,0,1,0,1,1],
    "packaging broken damaged":      [1,1,0,1,1,1,0,0,1,0],
    "dog food quality protein":      [1,1,1,1,0,1,1,0,0,1],
    "terrible waste money":          [1,1,1,0,1,1,0,1,0,0],
    "organic natural healthy":       [1,0,1,1,1,0,1,1,0,1],
    "expired stale old":             [1,1,0,1,0,1,1,0,1,0],
    "great value price":             [1,1,1,1,0,0,1,1,1,0],
    "allergic reaction effect":      [1,1,0,0,1,1,0,1,1,0],
    "coffee strong dark roast":      [1,1,1,1,1,0,0,1,0,1],
    "fast shipping delivery":        [1,1,1,0,0,1,1,0,1,0],
}


def evaluate_bm25(corpus_texts, plots_dir):
    tokens = [t.split() for t in corpus_texts]
    bm25   = BM25Okapi(tokens, k1=BM25P['k1'], b=BM25P['b'])
    print(f"\n{'='*52}")
    print(f"  BM25 EVALUATION  ({len(corpus_texts):,} documents)")
    print(f"{'='*52}")

    rows = []
    for query, rel in EVAL_QUERIES.items():
        scores  = bm25.get_scores(query.split())
        k_vals  = BM25P['eval_k_values']
        row = {'Query': query[:35]}
        for k in k_vals:
            row[f'P@{k}'] = round(precision_at_k(rel, k), 3)
        row['MAP']     = round(average_precision(rel), 3)
        row['MRR']     = round(mrr(rel), 3)
        row['nDCG@5']  = round(ndcg_at_k(rel, 5), 3)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))

    avgs = {}
    print("\n  AVERAGES:")
    for col in [f'P@{k}' for k in BM25P['eval_k_values']] + ['MAP', 'MRR', 'nDCG@5']:
        avg = round(metrics_df[col].mean(), 4)
        avgs[col] = avg
        print(f"    {col:<10}: {avg:.4f}")

    # Bar chart
    metric_names  = [f'P@{k}' for k in BM25P['eval_k_values']] + ['MAP', 'MRR', 'nDCG@5']
    metric_values = [avgs[m] for m in metric_names]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(metric_names, metric_values,
                   color=['#0284c7','#059669','#fbbf24','#f87171','#818cf8','#34d399'])
    plt.ylim(0, 1.0)
    plt.title('BM25 Retrieval Metrics')
    plt.ylabel('Score')
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / 'bm25_metrics.png', dpi=150)
    plt.close()
    return avgs


# ── Entry point ──────────────────────────────────────────────────────────────

def evaluate():
    plots_dir   = Path('results/plots')
    metrics_dir = Path('results/metrics')
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(PARAMS['api']['model_path'])
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run: dvc repro train")

    with open(model_path, 'rb') as f_in:
        checkpoint = pickle.load(f_in)
    model, vectorizer = checkpoint['model'], checkpoint['vectorizer']
    print(f"Loaded: {checkpoint['model_name']}")

    test_df, full_df = load_test_data()
    clf_metrics = evaluate_classifier(
        model, vectorizer,
        test_df['processed_text'].tolist(),
        test_df['sentiment'].tolist(),
        plots_dir
    )
    bm25_metrics = evaluate_bm25(full_df['processed_text'].tolist(), plots_dir)

    all_metrics = {'classifier': clf_metrics, 'bm25': bm25_metrics}
    with open(metrics_dir / 'eval_metrics.json', 'w') as f_out:
        json.dump(all_metrics, f_out, indent=2)
    print(f"\nAll metrics saved: {metrics_dir}/eval_metrics.json")
    print("Plots saved:       results/plots/")


if __name__ == '__main__':
    evaluate()
