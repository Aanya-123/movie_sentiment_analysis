import streamlit as st, joblib, re

@st.cache_resource
def load():
    return joblib.load("sentiment_model.joblib")

model = load()
st.title("🎬 Movie Review Sentiment Analyzer")
text = st.text_area("Paste a movie review", height=180)

if st.button("Analyze") and text.strip():
    clean = re.sub(r"<.*?>", " ", text).lower()
    p = model.predict_proba([clean])[0][1]
    st.subheader("Positive 😊" if p >= 0.5 else "Negative 😞")
    st.progress(float(p))
    st.caption(f"Positive probability: {p:.1%}")