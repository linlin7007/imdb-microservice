# IMDB Sentiment Microservice

This project deploys a Flask microservice that exposes a sentiment analysis model trained on the IMDB Movie Reviews dataset.

## Endpoints
- `GET /` - basic welcome message
- `GET /health` - health check
- `POST /predict` - sentiment prediction

## Example request
```bash
curl -X POST https://YOUR-RENDER-URL.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was amazing and I really enjoyed it\"}"
```

## Example response
```json
{
  "prediction": "positive",
  "confidence": 0.9321,
  "cleaned_text": "This movie was amazing and I really enjoyed it"
}
```

## Local run
```bash
pip install -r requirements.txt
python app.py
```

## Train the model
Place `IMDB Dataset.csv` in the same folder, then run:
```bash
python train_model.py
```
