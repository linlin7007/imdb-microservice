import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "IMDB Dataset.csv"
MODEL_PATH = "imdb_sentiment_model.pkl"

def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text))
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(DATA_PATH)
df["clean_review"] = df["review"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))),
    ("clf", SGDClassifier(loss="log_loss", random_state=42)),
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, pred)

joblib.dump(pipeline, MODEL_PATH)
print(f"Saved {MODEL_PATH}")
print(f"Test accuracy: {acc:.4f}")
