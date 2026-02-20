# nlp_utils.py
# Utilities: language detect, translate (fallback), sentiment, tone-aware paraphrase & summarization
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# quick translation fallback (unofficial)
from googletrans import Translator
translator = Translator()

# caches for lazy-loaded models/pipelines
_pipes = {
    "sentiment": None,
    "summarizer": None,
    "paraphrase_model": None,
    "paraphrase_tokenizer": None
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def translate_to_english(text: str) -> str:
    """Translate text to English using googletrans as fallback."""
    try:
        res = translator.translate(text, src='auto', dest='en')
        return res.text
    except Exception as e:
        # if translation fails, return original text
        print("translate error:", e)
        return text

# --------------------------
# Sentiment
# --------------------------
def _get_sentiment_pipe():
    if _pipes["sentiment"] is None:
        from transformers import pipeline
        # multilingual sentiment pipeline (reasonable tradeoff)
        _pipes["sentiment"] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    return _pipes["sentiment"]

def get_sentiment_label(text: str) -> str:
    """Return coarse sentiment: positive / negative / neutral"""
    try:
        pipe = _get_sentiment_pipe()
        out = pipe(text[:512])
        label = out[0].get("label", "").lower()
        # normalize label names to simple categories
        if "positive" in label or "pos" in label or label.startswith("4") or label.startswith("5"):
            return "positive"
        if "negative" in label or "neg" in label or label.startswith("1") or label.startswith("2"):
            return "negative"
        return "neutral"
    except Exception as e:
        print("sentiment error:", e)
        return "neutral"

# --------------------------
# Paraphrase (tone control)
# --------------------------
def _get_paraphrase_model_and_tokenizer(model_name="t5-small"):
    if _pipes["paraphrase_model"] is None or _pipes.get("paraphrase_model_name") != model_name:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _pipes["paraphrase_model"] = model
        _pipes["paraphrase_tokenizer"] = tokenizer
        _pipes["paraphrase_model_name"] = model_name
    return _pipes["paraphrase_model"], _pipes["paraphrase_tokenizer"]

def paraphrase_with_tone(text: str, tone: str = "neutral", model_name="t5-small"):
    """
    Paraphrase input text to match a tone.
    tone examples: 'neutral', 'positive', 'negative', 'formal', 'informal'
    """
    try:
        model, tokenizer = _get_paraphrase_model_and_tokenizer(model_name=model_name)
        # craft a short instruction prompt that many seq2seq models understand
        prompt = f"paraphrase in a {tone} tone: {text}"
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        gen = model.generate(
            inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        return out
    except Exception as e:
        print("paraphrase error:", e)
        return text

# --------------------------
# Tone-aware summarization
# --------------------------
def _get_summarizer(model_name="t5-small"):
    if _pipes["summarizer"] is None or _pipes.get("summ_model_name") != model_name:
        from transformers import pipeline
        _pipes["summarizer"] = pipeline("summarization", model=model_name, tokenizer=model_name)
        _pipes["summ_model_name"] = model_name
    return _pipes["summarizer"]

def summarize_with_tone(text: str, target_tone: str = "neutral",
                        model_name="t5-small", min_length=12, max_length=60,
                        do_paraphrase_before=True, do_paraphrase_after=False):
    """
    Steps:
     1) translate to English if needed
     2) optionally paraphrase to desired tone (preprocessing)
     3) summarize (beam search via pipeline)
     4) optionally paraphrase summary to enforce tone
    model_name: "t5-small" (fast) or "t5-base"/"t5-large"/"facebook/bart-large-cnn" (higher quality)
    """
    try:
        # 1) Translate to English if input not English
        from langdetect import detect as _detect
        if _detect(text) != 'en':
            text_eng = translate_to_english(text)
        else:
            text_eng = text

        # 2) Paraphrase before summarization to nudge tone (optional)
        if do_paraphrase_before and target_tone and target_tone != "neutral":
            try:
                text_eng = paraphrase_with_tone(text_eng, tone=target_tone, model_name=model_name)
            except Exception:
                pass

        # 3) Summarize
        summarizer = _get_summarizer(model_name=model_name)
        summary_list = summarizer(
            text_eng,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        summary = summary_list[0]["summary_text"] if summary_list else ""

        # 4) Paraphrase summary after to strengthen tone (optional)
        if do_paraphrase_after and target_tone and target_tone != "neutral":
            try:
                summary = paraphrase_with_tone(summary, tone=target_tone, model_name=model_name)
            except Exception:
                pass

        # Extra brevity: shorten if still long
        if len(summary.split()) > 50:
            sentences = summary.split('. ')
            summary = '. '.join(sentences[:2]).strip()
            if not summary.endswith('.'):
                summary += '.'
        return summary
    except Exception as e:
        print("summarize_with_tone error:", e)
        return ""