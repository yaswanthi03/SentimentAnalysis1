import streamlit as st
import joblib
import re
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.write(
    """
    This application predicts sentiment of Amazon product reviews
    using a Machine Learning model trained on **500,000+ reviews**.
    
    **Sentiments:**
    - Positive 😊
    - Neutral 😐
    - Negative 😞
    """
)

st.sidebar.markdown("---")
st.sidebar.write("Built using **Python, NLP, ML & Streamlit**")

# Main title
st.markdown(
    "<h1 style='text-align: center;'>🛍️ Amazon Review Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.write("")
st.write("Enter a product review below and click **Analyze**.")

# Input box
user_input = st.text_area(
    "✍️ Your Review",
    height=150,
    placeholder="Example: The product quality is amazing and delivery was fast!"
)

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        # Confidence score
        probs = model.predict_proba(vectorized)[0]
        confidence = np.max(probs) * 100

        st.markdown("### 📊 Result")

        if prediction == "positive":
            st.success(f"😊 **Positive Review**")
        elif prediction == "negative":
            st.error(f"😞 **Negative Review**")
        else:
            st.info(f"😐 **Neutral Review**")

        st.write(f"**Confidence:** {confidence:.2f}%")

        st.markdown("---")
        st.caption("Model: Logistic Regression with TF-IDF features")
