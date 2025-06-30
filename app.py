import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os

# Title and description
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis Web App")
st.write("Analyze your text using different sentiment analysis models and generate a word cloud.")

# CSV File Setup
CSV_FILE = "feedback.csv"

def initialize_csv(file_path):
    # Create or reset the CSV with proper headers if empty or non-existent
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        df = pd.DataFrame(columns=["Text", "Sentiment", "Model"])
        df.to_csv(file_path, index=False)

initialize_csv(CSV_FILE)

# Load transformer model
@st.cache_resource
def load_transformer():
    return pipeline("sentiment-analysis")

transformer_model = load_transformer()

# Sidebar for model selection
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Sentiment Model", ["VADER", "TextBlob", "Transformer"])

# Text input
user_input = st.text_area("Enter your sentence or paragraph:")

# WordCloud generator
def generate_wordcloud(text, sentiment_label):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.subheader(f"{sentiment_label} Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Analyze button
if st.button("Analyze"):
    if user_input.strip():
        sentiment_label = "Neutral"
        compound = 0.0

        if model_choice == "VADER":
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(user_input)
            compound = score["compound"]
            if compound >= 0.05:
                sentiment_label = "Positive"
            elif compound <= -0.05:
                sentiment_label = "Negative"

        elif model_choice == "TextBlob":
            blob = TextBlob(user_input)
            compound = blob.sentiment.polarity
            if compound > 0.05:
                sentiment_label = "Positive"
            elif compound < -0.05:
                sentiment_label = "Negative"

        elif model_choice == "Transformer":
            result = transformer_model(user_input)[0]
            sentiment_label = result["label"].capitalize()
            compound = result["score"] if result["label"] == "POSITIVE" else -result["score"]

        # Display result
        st.markdown(f"### 🧾 Sentiment: `{sentiment_label}` (Score: `{compound:.2f}`)")
        generate_wordcloud(user_input, sentiment_label)

        # Store feedback
        new_entry = pd.DataFrame([[user_input, sentiment_label, model_choice]], columns=["Text", "Sentiment", "Model"])
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
    else:
        st.warning("Please enter some text.")

# View feedback table
with st.expander("📊 View Feedback Data"):
    df = pd.read_csv(CSV_FILE)
    st.dataframe(df)
