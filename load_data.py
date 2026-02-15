import pandas as pd
import re

print("Loading data...")

df = pd.read_csv("data/reviews.csv")
df = df[["Score", "Text"]]

# Convert Score to Sentiment
def convert_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df["Sentiment"] = df["Score"].apply(convert_sentiment)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()                     # lowercase
    text = re.sub(r"[^a-z\s]", "", text)    # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df["Cleaned_Text"] = df["Text"].apply(clean_text)

print(df[["Text", "Cleaned_Text"]].head())
print("Done cleaning.")
