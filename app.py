# app.py
import streamlit as st
from nlp_utils import (
    detect_language,
    translate_to_english,
    get_sentiment_label,
    summarize_with_tone
)

st.set_page_config(page_title="Multilingual NLP — Tone Aware", layout="centered")
st.title("Multilingual NLP — Tone-aware Summarizer")
st.markdown("Enter text in any language. Choose a tone for the summary. Models download on first use (may take a few minutes).")

text = st.text_area("Enter text here", height=200)
task = st.radio("Choose task:", ["All", "Translate only", "Sentiment only", "Summarize only"])
tone = st.selectbox("Desired tone for summary/translation:", ["neutral", "positive", "negative", "formal", "informal"])
model_choice = st.selectbox("Model for summarization (quality vs speed):", ["t5-small (fast)", "t5-base", "facebook/bart-large-cnn (better)"])

# map to actual model names
MODEL_MAP = {
    "t5-small (fast)": "t5-small",
    "t5-base": "t5-base",
    "facebook/bart-large-cnn (better)": "facebook/bart-large-cnn"
}

if st.button("Run"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        # Detect language
        lang = detect_language(text)
        st.info(f"Detected language code: *{lang}*")

        if task in ["All", "Translate only", "Summarize only"]:
            with st.spinner("Translating (if needed)..."):
                translated = translate_to_english(text)
            st.subheader("Translation (to English)")
            st.write(translated)

        if task in ["All", "Sentiment only"]:
            with st.spinner("Detecting sentiment..."):
                # use translated text to detect sentiment (more reliable)
                translated_for_sent = translate_to_english(text)
                sentiment = get_sentiment_label(translated_for_sent)
            st.subheader("Detected Sentiment (coarse)")
            st.write(sentiment)

        if task in ["All", "Summarize only"]:
            with st.spinner("Generating tone-aware summary (this may take time on first run)..."):
                model_name = MODEL_MAP.get(model_choice, "t5-small")
                # parameters: adjust min_length/max_length if you want briefer/longer summaries
                summary = summarize_with_tone(
                    text,
                    target_tone=tone,
                    model_name=model_name,
                    min_length=12,
                    max_length=50,
                    do_paraphrase_before=True,
                    do_paraphrase_after=False
                )
            st.subheader("Tone-aware Summary")
            if summary:
                st.write(summary)
            else:
                st.write("Summary could not be generated.")