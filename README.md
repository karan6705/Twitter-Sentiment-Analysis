# Twitter Sentiment Analysis Capstone

A reproducible end-to-end pipeline for classifying tweet sentiment using Logistic Regression and bag-of-words features.

## Table of Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Conclusion & Next Steps](#conclusion--next-steps)

---

## Installation

Install the required dependencies and clone the repository:

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
```

## Data

The project uses a CSV file with 31,962 labeled tweets.

* **Columns:**

  * `id`: Unique identifier
  * `label`: Sentiment (0 = negative, 1 = positive)
  * `tweet`: Raw tweet text

## Methodology

1. **Data Loading:** Read the CSV into a pandas DataFrame.
2. **Exploratory Data Analysis:** Inspect dataset shape, distributions, and sample tweets.
3. **Text Cleaning:** Remove Twitter handles, URLs, non-alphanumeric characters, and lowercase.
4. **Tokenization & Stemming:** Split into tokens and apply Porter Stemmer to reduce inflection.
5. **Feature Extraction:** Convert text to a bag-of-words matrix (1,000 features) with `CountVectorizer`.
6. **Model Training:** Train a `LogisticRegression` classifier on a 75/25 train/test split.
7. **Evaluation:** Compute accuracy, F1-score, confusion matrix, ROC-AUC, and precision-recall curves.

## Results

* **Accuracy:** 0.95
* **F1 Score:** 0.56
* **ROC AUC:** 0.72


## Usage

Import the helper and call `predict_sentiment` on new text:

```python
from sentiment import predict_sentiment

print(predict_sentiment("I love this product!"))   # Positive
print(predict_sentiment("Worst experience ever.")) # Negative
```

