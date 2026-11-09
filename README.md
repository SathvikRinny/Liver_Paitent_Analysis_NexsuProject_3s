# Liver Health Check: Liver Disease Risk Prediction

A Flask web app that estimates the risk of liver disease from ten values on a routine liver function test. It uses a machine-learning model trained on the Indian Liver Patient Dataset. Started as a Nexus internship project and since rebuilt.

## Features

- Clean, responsive interface (works on phones)
- Form with units, typical ranges, server-side validation and clear error messages
- Result page with a **risk percentage gauge** (low / moderate / high) instead of a plain yes/no
- Flags every test value that is **outside the typical adult range**
- JSON API at `/api/predict`
- Reproducible training script with model comparison and saved metrics
- Automated tests (`pytest`)

## Model

| | |
|---|---|
| Dataset | Indian Liver Patient Dataset, 583 patients (`data/indian_liver_patient.csv`) |
| Preprocessing | Median imputation, log transform of skewed lab values, standard scaling |
| Candidates | Logistic regression, random forest, gradient boosting (5-fold CV) |
| Selected | Random forest (class-balanced) |
| Test ROC AUC | 0.78 |
| Test accuracy | 69% |
| Recall (liver patients found) | 76% |

Full numbers are in `model/metrics.json`. To retrain:

```bash
python train.py
```

### Input features

| Feature | Unit | Typical range |
|---|---|---|
| Age | years | |
| Gender | male / female | |
| Total bilirubin | mg/dL | 0.1–1.2 |
| Direct bilirubin | mg/dL | 0–0.3 |
| Alkaline phosphatase (ALP) | IU/L | 44–147 |
| ALT (SGPT) | IU/L | 7–56 |
| AST (SGOT) | IU/L | 10–40 |
| Total proteins | g/dL | 6.0–8.3 |
| Albumin | g/dL | 3.5–5.0 |
| Albumin/globulin ratio | | 1.1–2.5 |

## Running locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Set `FLASK_DEBUG=1` for auto-reload while developing.

For production: `gunicorn app:app`

## API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender":"male","Age":45,"Total_Bilirubin":0.9,"Direct_Bilirubin":0.2,
       "Alkaline_Phosphotase":180,"Alamine_Aminotransferase":35,
       "Aspartate_Aminotransferase":38,"Total_Protiens":6.9,"Albumin":3.4,
       "Albumin_and_Globulin_Ratio":1.0}'
```

Returns `{"probability": 0.44, "risk": "moderate", "out_of_range": [...]}`, or `400` with an `errors` object.

## Tests

```bash
pip install pytest
pytest
```

## Project structure

```
app.py                Flask app, validation and prediction
train.py              Model training and evaluation
model/                Trained model and metrics
data/                 Dataset
templates/            HTML pages (Jinja2)
static/               CSS and images
tests/                pytest tests
```

## What changed from the original version

- Fixed the result page always saying "No worries" because of a typo (`predition`)
- Fixed gender being encoded as 0 for everyone during training (`'male'` vs `'Male'`)
- Replaced the unscaled SVM (which flagged almost everyone) with a properly preprocessed, cross-validated model
- Moved pages into `templates/` and `static/` so Flask can actually find them

## Team

P. Sathvik Raju, R. Varun, S. Sai Revanth, H. N. Vinay Reddy

## Disclaimer

This is a student project for learning purposes. It is not a medical device and does not give a diagnosis. Always consult a doctor.
