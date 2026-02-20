# sentiment.py
import csv
from nlp_utils import analyze_sentiment

texts = []
labels = []
with open("sample_data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        texts.append(r["text"])
        labels.append(r["lang_label"])  # only if sentiment labels exist; adjust fields accordingly

# This is a placeholder: you'd need sample_data.csv with sentiment labels to evaluate sentiment
for t in texts:
    print("Text:", t)
    print("Sentiment:", analyze_sentiment(t))
    print("-" * 40)