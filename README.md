# Amazon Reviews Sentiment Analysis

## Project Overview
This project analyzes customer sentiment from Amazon product reviews using Natural Language Processing (NLP) and Machine Learning.

A Logistic Regression model trained on over 500,000 reviews predicts whether a review is **Positive**, **Negative**, or **Neutral**.

An interactive Streamlit web application allows real-time sentiment prediction.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- VS Code

---

## Project Workflow
1. Data loading and preprocessing
2. Text cleaning and normalization
3. TF-IDF feature extraction
4. Model training and evaluation
5. Model persistence
6. Streamlit dashboard deployment

---

## Model Performance
- Accuracy: **87%**
- Strong performance on positive and negative reviews
- Neutral class handled with class imbalance awareness

---

## How to Run
```bash
pip install -r requirements.txt
python src/model.py
streamlit run app.py
