# NLP Sentiment Analysis — Amazon Product Reviews

**Tools:** Python, NLTK, Scikit-learn, Seaborn, WordCloud
**Domain:** Natural Language Processing / E-Commerce

---

Text classification at the review level is a well-worn problem, but the interesting part is always in the errors — which reviews does the model get wrong, and why? This project builds a sentiment pipeline on 10,000+ Amazon product reviews (positive, negative, neutral) and compares Logistic Regression against Linear SVM to see where they diverge.

TF-IDF with unigrams and bigrams turned out to be the key feature engineering decision. Bigrams like "highly recommend" and "broke after" carry far more signal than individual words, and they pushed both models noticeably higher than unigram-only baselines.

---

## Results

| Model | F1-Score |
|---|---|
| Logistic Regression | ~0.91 |
| Linear SVM | ~0.91 |

Both models landed at the same F1, but they made different mistakes. Negative reviews were consistently easier to classify correctly — the language tends to be more extreme and distinctive. Neutral reviews were the hardest class; they sit in the middle of the feature space and bleed into both positive and negative predictions.

The word cloud for positive reviews surfaces phrases you'd expect ("great product", "works perfectly"), but also a handful of hedged positives ("pretty good", "does the job") that the model sometimes misclassifies as neutral. That's a real signal about customer language patterns, not just a visualization artifact.

---

## Project structure

```
nlp-sentiment-analysis/
├── sentiment_analysis.py     # full NLP pipeline: cleaning, TF-IDF, modeling, eval
├── sentiment_analysis.png    # confusion matrix + model comparison + word cloud
├── nlp_requirements.txt
└── README.md
```

---

## Running it

```bash
pip install -r nlp_requirements.txt
python sentiment_analysis.py
```

Trains both models, prints classification reports, and saves the output chart.

---

## Skills demonstrated

Text preprocessing · TF-IDF vectorization · Logistic Regression · Linear SVM · F1-score evaluation · WordCloud · NLTK
