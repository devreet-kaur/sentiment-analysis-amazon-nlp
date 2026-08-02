# Model Card - Amazon Fine Food Reviews Sentiment Classifier

## Model Details

| Field | Value |
|-------|-------|
| Model name | LogReg + TF-IDF Sentiment Classifier |
| Version | 1.0.0 |
| Type | Binary text classification |
| Framework | scikit-learn 1.5.1 |
| Course | MAI203 - Introduction to Natural Language Processing, Seneca Polytechnic |
| Instructor | Prof. Junwei Huang |
| Team | Devreet, Arushi |
| Date | 2025 |

## Intended Use

This model predicts whether an Amazon food product review expresses positive or negative sentiment. It is intended for academic demonstration of NLP pipeline construction (preprocessing, traditional ML, transformer inference, information retrieval) in the context of MAI203.

Do not use this model in production systems for real purchasing decisions or content moderation without further validation and fine-tuning.

## Architecture

**Primary model:** Logistic Regression with L2 regularization  
**Feature extraction:** TF-IDF (max 20,000 features, unigrams and bigrams, sublinear TF scaling)  
**Class handling:** class_weight='balanced' to account for the positive-skewed class distribution  
**Baseline model:** Multinomial Naive Bayes with BoW features  
**Advanced inference:** DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) on a 2,000-row sample  

## Training Data

| Field | Value |
|-------|-------|
| Dataset | Amazon Fine Food Reviews |
| Source | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews |
| Full size | 568,454 reviews |
| Sample used | 50,000 rows (random_state=42) |
| Label mapping | Score 4-5 = positive, Score 1-2 = negative, Score 3 = dropped |
| Train split | 70% |
| Val split | 15% |
| Test split | 15% |

## Preprocessing Pipeline

1. Lowercase all text
2. Strip HTML tags (reviews contain br tags from Amazon's editor)
3. Remove URLs
4. Remove punctuation and numbers (letters only)
5. Tokenize with NLTK word_tokenize
6. Remove English stopwords
7. Lemmatize with WordNetLemmatizer using POS-tag-aware lookup

## Hyperparameters

All hyperparameters are stored in params.yaml. Key values:

| Parameter | Value |
|-----------|-------|
| tfidf.max_features | 20,000 |
| tfidf.ngram_range | (1, 2) |
| tfidf.min_df | 3 |
| tfidf.sublinear_tf | true |
| logreg.C | [fill after grid search] |
| logreg.class_weight | balanced |
| logreg.max_iter | 1,000 |
| logreg.solver | lbfgs |

## Evaluation Metrics

### Sentiment Classifier (Part B)

| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| VADER (lexicon) | [fill] | [fill] | Lexicon baseline, no training |
| LogReg + TF-IDF | [fill] | [fill] | Primary model |
| DistilBERT (pretrained) | [fill] | [fill] | SST-2 fine-tuned, 2k sample |

### Text Classifier (Part C - 3 classes)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Naive Bayes + BoW | [fill] | [fill] |
| LogReg + TF-IDF | [fill] | [fill] |

### BM25 Retrieval (Part D)

| Metric | Value |
|--------|-------|
| MAP | [fill] |
| MRR | [fill] |
| P@5 | [fill] |
| P@10 | [fill] |
| nDCG@5 | [fill] |

## Limitations

**Lexicon gap:** The model was trained on Amazon food reviews only. Words like "sick" (positive slang in food context) may be misclassified because they carry negative connotation in general-domain lexicons such as VADER.

**Language coverage:** Preprocessing and stop words cover English only. Reviews written in other languages will produce noise tokens rather than meaningful features.

**Fake review bias:** Amazon has documented problems with incentivized and fake reviews. A model trained on this data learns patterns from both genuine and manufactured sentiment, which may not reflect actual product quality.

**Class imbalance:** Roughly 80% of the original dataset is 4-5 star reviews. Without class_weight='balanced', the model would predict "positive" for nearly all inputs.

**Sarcasm:** Neither TF-IDF nor DistilBERT (pretrained on SST-2) reliably detects sarcasm in food reviews. "Oh great, another product that breaks in a week" would likely be predicted as positive.

## Ethical Considerations

Deploying a food review sentiment classifier at scale could be used to suppress negative reviews, selectively surface positive ones, or generate misleading product quality signals. Transparent disclosure of how search results and sentiment scores are computed is required for responsible use.

User review data contains personal product preferences that may be sensitive. The Amazon Fine Food Reviews dataset is publicly available, but any system that collects new reviews should comply with applicable privacy regulations.
