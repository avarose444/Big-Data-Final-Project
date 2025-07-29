# 📰 Fake vs Real News Classifier — NLP & Deep Learning

## 📌 Project Overview

This project explores supervised learning and deep learning techniques to classify news articles as real or fake. Using a labeled Kaggle dataset of 44,000 articles, we applied natural language processing techniques, engineered features based on readability and sentiment, and evaluated a variety of models including LSTM and XGBoost.

The pipeline includes:
- 🧹 Text preprocessing: cleaning, tokenization, padding
- 🧠 Feature engineering (e.g. SMOG Index, subjectivity, sentiment)
- 🧾 TF-IDF vectorization for sparse feature extraction
- ⚙️ Multiple model comparisons: Logistic Regression, XGBoost, LSTM
- 🧪 Evaluation using accuracy, F1 score, and interpretability of features

The goal is to develop a reliable classification system for detecting misinformation using linguistic and semantic patterns in article content.

---

## 🛠️ Key Tools & Techniques

- Python (`pandas`, `numpy`)
- NLP preprocessing: `NLTK`, `re`, `string`, `textblob`
- Machine Learning: `scikit-learn`, `XGBoost`
- Deep Learning: `TensorFlow`, `Keras` (LSTM models)
- Evaluation: Confusion Matrix, F1 Score, Accuracy

---

## 📂 Dataset

- **Source**: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- 44,000 articles in two CSV files (`Fake.csv`, `True.csv`)
- Features: `title`, `text`, `subject`, `date`
- Label: 1 (Real), 0 (Fake)

---

## 📈 Summary of Results

| Model                    | Accuracy | F1 Score |
|--------------------------|----------|----------|
| Logistic Regression      | 71%      | 0.69     |
| XGBoost                  | 80%      | 0.79     |
| LSTM                     | 93%      | 0.90     |
| Logistic Regression + LSTM | 97%   | 0.97     |
| XGBoost + LSTM           | **98%**  | **0.98** |

- Fake news articles were more **subjective** and **linguistically complex**
- Real news was **simpler** and had higher **readability**
- LSTM models dramatically improved classification performance

---

## 🧠 Feature Engineering

We extracted statistical, semantic, and structural features:
- SMOG Index, Coleman-Liau Index
- Sentence and word length
- Subjectivity and sentiment scores
- TF-IDF vectors for key terms (e.g. “Reuters” was a strong real-news signal)

---

## 🔬 Evaluation Techniques

- Train/test split (80/20)
- Confusion matrix analysis
- Precision, recall, and F1 score
- Manual inspection of misclassified articles

---

## 🚀 Usage (Coming Soon)

```bash
git clone https://github.com/lsgordon/Big-Data-Final-Project
cd Big-Data-Final-Project
python run_model.py
