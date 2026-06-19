# 💬 NLP Sentiment Analysis — Amazon Product Reviews

**Author:** Reddy Shreya  
**Tools:** Python · NLTK · Scikit-learn · Seaborn · WordCloud  
**Domain:** Natural Language Processing / E-Commerce

---

## 🎯 Project Overview

This project applies NLP techniques to classify sentiment in **10,000+ product reviews** as positive, negative, or neutral.  
It compares Logistic Regression and Linear SVM classifiers using TF-IDF features.

---

## 📁 Project Structure

```
nlp-sentiment-analysis/
│
├── sentiment_analysis.py    # Full NLP pipeline
├── sentiment_analysis.png   # Confusion matrix + model comparison + word cloud
├── requirements.txt
└── README.md
```

---

## 🔍 Pipeline Steps

| Step | Description |
|------|-------------|
| Data Simulation | 10,000 labeled product reviews (positive/negative/neutral) |
| Text Cleaning | Lowercasing, punctuation removal, whitespace normalization |
| Feature Extraction | TF-IDF vectorization (5,000 features, unigrams + bigrams) |
| Modeling | Logistic Regression vs. Linear SVM |
| Evaluation | Classification report, F1-score, confusion matrix |
| Visualization | Word cloud for positive reviews, model comparison chart |

---

## 📈 Results

| Model | F1-Score |
|-------|----------|
| Logistic Regression | ~0.91 |
| Linear SVM | ~0.91 |

- **Best Feature:** TF-IDF bigrams capture phrases like "highly recommend" and "broke after"
- **Insight:** Negative reviews use more extreme language, making them easier to classify correctly

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

---

## 🛠️ Requirements

See `requirements.txt`
