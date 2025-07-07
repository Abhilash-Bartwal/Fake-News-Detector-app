# fake_news_app.py
import streamlit as st
import joblib
import re

# Load
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("fake_news_vectorizer.pkl")

# Clean input
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()

st.set_page_config(page_title="Fake News Detector")
st.title("📰 Fake News Detection App")

st.markdown("Enter a news article or headline below:")

user_input = st.text_area("News Text", height=200)

if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some content.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.decision_function(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ Real News (Confidence: {confidence:.2f})")
        else:
            st.error(f"🚫 Fake News (Confidence: {confidence:.2f})")

        st.caption("Note: Higher confidence means the model is more sure about its prediction.")
