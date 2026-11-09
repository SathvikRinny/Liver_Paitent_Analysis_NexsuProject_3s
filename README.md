# Liver Patient Analysis and Prediction

A machine-learning web app that predicts whether a patient is likely to have liver disease from routine blood-test values. Built with Python, scikit-learn and Flask as part of an internship project.

## How it works

1. A classification model was trained on the **Indian Liver Patient Dataset** (583 patient records, 10 features) and saved as `liver_analysis.pkl`.
2. The Flask app (`app.py`) loads the model and serves three pages:
   - `p1.html`: introduction to the problem
   - `p2.html`: form where the user enters the test values
   - `p3.html`: shows the prediction result

### Input features

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Gender | 1 = male, 0 = female |
| Total_Bilirubin | Total bilirubin (mg/dL) |
| Direct_Bilirubin | Direct bilirubin (mg/dL) |
| Alkaline_Phosphotase | Alkaline phosphatase (IU/L) |
| Alamine_Aminotransferase | ALT (IU/L) |
| Aspartate_Aminotransferase | AST (IU/L) |
| Total_Protiens | Total proteins (g/dL) |
| Albumin | Albumin (g/dL) |
| Albumin_and_Globulin_Ratio | A/G ratio |

## Running locally

Flask looks for HTML pages in a `templates/` folder and for CSS/images in `static/`. The files in this repo are stored flat, so arrange them first:

```bash
mkdir -p templates static/css
mv p1.html p2.html p3.html templates/
mv styles.css ptwo.css va1.jpeg doc1.jpeg static/css/

pip install flask numpy scikit-learn
python app.py
```

Then open http://127.0.0.1:5000.

## Tech stack

Python, Flask, scikit-learn, NumPy, HTML/CSS

## Disclaimer

This is a student project for learning purposes. It is not a medical tool; always consult a doctor.
