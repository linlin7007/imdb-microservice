import os
import re
import joblib
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "imdb_sentiment_model.pkl")

app = Flask(__name__)

def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text))
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    model_load_error = str(e)
else:
    model_load_error = None

@app.get("/")
def home():
    return jsonify({
        "message": "IMDB Sentiment Analysis API is running.",
        "healthcheck": "/health",
        "prediction_endpoint": "/predict",
        "example_input": {"text": "This movie was amazing and I really enjoyed it"}
    })

@app.get("/health")
def health():
    if model is None:
        return jsonify({"status": "error", "model_loaded": False, "detail": model_load_error}), 500
    return jsonify({"status": "ok", "model_loaded": True})

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load.", "detail": model_load_error}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": 'Request body must be JSON and include a "text" field.'}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": '"text" must not be empty.'}), 400

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
        confidence = float(probs[classes.index(prediction)])
    elif hasattr(model, "decision_function"):
        import math
        score = model.decision_function([cleaned])[0]
        # Binary logistic-style approximation for nice API output
        if isinstance(score, (list, tuple)):
            score = score[0]
        if prediction == "positive":
            confidence = 1 / (1 + math.exp(-float(score)))
        else:
            confidence = 1 - (1 / (1 + math.exp(-float(score))))
        confidence = float(confidence)

    return jsonify({
        "prediction": str(prediction),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "cleaned_text": cleaned
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
