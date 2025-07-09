<p align="center">
  <img src="https://abs.twimg.com/icons/apple-touch-icon-192x192.png" alt="X Logo" width="96" height="96" />
</p>

<h1 align="center">🐦 Twitter Sentiment Analysis Capstone</h1>

<p align="center">
  [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-000000?style=for-the-badge&logo=python)](https://www.python.org/)
  [![License MIT](https://img.shields.io/badge/License-MIT-1DA1F2?style=for-the-badge)](LICENSE)
</p>

---

This project provides a reproducible end-to-end pipeline for classifying tweet sentiment using a Logistic Regression model and bag‑of‑words features.

## 🐦 Table of Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Conclusion & Next Steps](#conclusion--next-steps)

---

## 🚀 Installation

Clone the repo and install dependencies:

```bash
# 1. Clone this repository
git clone https://github.com/karan6705/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

# 2. (Optional) create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## 🗄️ Data

This analysis uses a CSV file with 31,962 labeled tweets.

* **Filename:** `Twitter Sentiments.csv`
* **Columns:**

  * `id` — Unique tweet identifier
  * `label` — Sentiment (0 = Negative, 1 = Positive)
  * `tweet` — Raw tweet text

Place the CSV in the project root before running the analysis.

---

## 🔍 Methodology

1. **Data Loading:** Read tweets CSV into a Pandas DataFrame.
2. **Exploratory Analysis:** Inspect dataset shape, sample tweets, and label distribution.
3. **Text Cleaning:** Remove Twitter handles (`@user`), URLs, non-alphanumeric chars, and lowercase text.
4. **Tokenization & Stemming:** Split text into tokens and apply NLTK’s PorterStemmer.
5. **Feature Extraction:** Vectorize tweets into a bag‑of‑words matrix (`CountVectorizer`, max 1000 features).
6. **Model Training:** Fit a `LogisticRegression` classifier on 75/25 train/test split.
7. **Evaluation:** Calculate accuracy, F1-score, ROC-AUC, and display confusion matrix & precision‑recall curves.

---

## 📊 Results

* **Accuracy:** 0.95
* **F1 Score:** 0.56
* **ROC AUC:** 0.72

Plots and metrics snapshots are available in the `reports/` folder.

---

## ▶️ Usage

After training, use the helper function to predict sentiment on new tweets:

```python
from sentiment import predict_sentiment

print(predict_sentiment("I love this product!"))   # Should output: Positive
print(predict_sentiment("Worst experience ever.")) # Should output: Negative
```

Model artifacts (`count_vect.joblib`, `logreg_model.joblib`) are saved in the `models/` directory.

---

## 📝 Conclusion & Next Steps

* The Logistic Regression model achieves strong overall accuracy but lower recall on positive tweets (F1 0.56).
* **Next steps:** experiment with additional features (bi‑grams, TF‑IDF), more complex models (Random Forest, XGBoost), and hyperparameter tuning.

---

## 📄 License

Released under the [MIT License](LICENSE).
