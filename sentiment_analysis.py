# ============================================================
# NLP Sentiment Analysis — Amazon Product Reviews
# Author: Reddy Shreya
# Tools: Python, NLTK, Scikit-learn, Seaborn, WordCloud
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import re, string
import warnings
warnings.filterwarnings('ignore')

# Optional: WordCloud (install separately if needed)
try:
    from wordcloud import WordCloud
    WORDCLOUD = True
except ImportError:
    WORDCLOUD = False
    print("WordCloud not installed — skipping word cloud plot. Run: pip install wordcloud")

# ── 1. GENERATE SIMULATED REVIEW DATASET ────────────────────
np.random.seed(42)
n = 10000

positive_phrases = [
    "absolutely love this product", "great quality and fast shipping",
    "exceeded my expectations", "would highly recommend", "perfect fit",
    "amazing value for the price", "works exactly as described",
    "very happy with this purchase", "excellent customer service",
    "beautiful packaging and high quality"
]
negative_phrases = [
    "terrible quality broke after one day", "complete waste of money",
    "do not buy this product", "very disappointed with purchase",
    "stopped working after a week", "nothing like the description",
    "worst purchase I have ever made", "cheap materials fell apart",
    "customer service was unhelpful", "arrived damaged and late"
]
neutral_phrases = [
    "product is okay nothing special", "average quality for the price",
    "it works but nothing outstanding", "decent product meets basic needs",
    "not bad but not great either"
]

sentiments, reviews = [], []
for _ in range(n):
    s = np.random.choice(['positive', 'negative', 'neutral'], p=[0.55, 0.30, 0.15])
    sentiments.append(s)
    if s == 'positive':
        base = np.random.choice(positive_phrases)
    elif s == 'negative':
        base = np.random.choice(negative_phrases)
    else:
        base = np.random.choice(neutral_phrases)
    noise = np.random.choice(['', ' really', ' truly', ' honestly', ' just'], p=[0.5, 0.15, 0.15, 0.1, 0.1])
    reviews.append(base + noise)

df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
print("Dataset Shape:", df.shape)
print("\nSentiment Distribution:\n", df['sentiment'].value_counts())

# ── 2. TEXT PREPROCESSING ────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Binary: positive vs negative (drop neutral for binary classifier)
df_binary = df[df['sentiment'] != 'neutral'].copy()
df_binary['label'] = (df_binary['sentiment'] == 'positive').astype(int)
print(f"\nBinary dataset size: {len(df_binary):,} reviews")

# ── 3. TF-IDF VECTORIZATION ──────────────────────────────────
X = df_binary['clean_review']
y = df_binary['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"\nTF-IDF feature matrix: {X_train_tfidf.shape}")

# ── 4. MODEL TRAINING ───────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, C=1.0, random_state=42),
    'Linear SVM':          LinearSVC(C=1.0, max_iter=2000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'model': model, 'preds': y_pred, 'f1': f1}
    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print(f"F1-Score: {f1:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
best_preds = results[best_name]['preds']
print(f"\nBest Model: {best_name} | F1 = {results[best_name]['f1']:.4f}")

# ── 5. VISUALIZATIONS ───────────────────────────────────────
cols = 3 if WORDCLOUD else 2
fig, axes = plt.subplots(1, cols, figsize=(6*cols, 5))
fig.suptitle('NLP Sentiment Analysis — Reddy Shreya', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[0].set_title(f'Confusion Matrix\n({best_name})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Model comparison bar
f1_scores = {k: v['f1'] for k, v in results.items()}
axes[1].bar(f1_scores.keys(), f1_scores.values(), color=['steelblue', '#e74c3c'], edgecolor='black')
axes[1].set_title('Model F1-Score Comparison')
axes[1].set_ylabel('F1-Score')
axes[1].set_ylim(0.8, 1.0)
for i, (k, v) in enumerate(f1_scores.items()):
    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

# WordCloud
if WORDCLOUD:
    pos_text = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
    wc = WordCloud(width=600, height=400, background_color='white', colormap='Blues').generate(pos_text)
    axes[2].imshow(wc, interpolation='bilinear')
    axes[2].axis('off')
    axes[2].set_title('WordCloud — Positive Reviews')

plt.tight_layout()
plt.savefig('sentiment_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as sentiment_analysis.png")
