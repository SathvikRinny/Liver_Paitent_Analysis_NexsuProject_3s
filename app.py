"""Liver disease risk prediction web app (Flask)."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
MODEL = joblib.load(ROOT / 'model' / 'liver_model.joblib')
METRICS = json.loads((ROOT / 'model' / 'metrics.json').read_text())

# name, label, unit, allowed min, allowed max, normal low, normal high, help text
FIELDS = [
    ('Age', 'Age', 'years', 1, 100, None, None, 'Patient age in years'),
    ('Total_Bilirubin', 'Total bilirubin', 'mg/dL', 0, 80, 0.1, 1.2, 'Typical: 0.1–1.2'),
    ('Direct_Bilirubin', 'Direct bilirubin', 'mg/dL', 0, 40, 0.0, 0.3, 'Typical: 0–0.3'),
    ('Alkaline_Phosphotase', 'Alkaline phosphatase (ALP)', 'IU/L', 10, 3000, 44, 147, 'Typical: 44–147'),
    ('Alamine_Aminotransferase', 'ALT (SGPT)', 'IU/L', 1, 3000, 7, 56, 'Typical: 7–56'),
    ('Aspartate_Aminotransferase', 'AST (SGOT)', 'IU/L', 1, 5000, 10, 40, 'Typical: 10–40'),
    ('Total_Protiens', 'Total proteins', 'g/dL', 1, 12, 6.0, 8.3, 'Typical: 6.0–8.3'),
    ('Albumin', 'Albumin', 'g/dL', 0.5, 7, 3.5, 5.0, 'Typical: 3.5–5.0'),
    ('Albumin_and_Globulin_Ratio', 'Albumin/globulin ratio', '', 0.1, 3.5, 1.1, 2.5, 'Typical: 1.1–2.5'),
]
FEATURE_ORDER = METRICS['features']

app = Flask(__name__)


def parse_input(form):
    """Validate the submitted values. Returns (values, errors)."""
    values, errors = {}, {}

    gender = str(form.get('Gender', '')).strip().lower()
    if gender in ('male', '1'):
        values['Gender'] = 1
    elif gender in ('female', '0'):
        values['Gender'] = 0
    else:
        errors['Gender'] = 'Please choose a gender.'

    for name, label, unit, lo, hi, *_ in FIELDS:
        raw = str(form.get(name, '')).strip()
        if raw == '':
            errors[name] = 'This field is required.'
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[name] = 'Please enter a number.'
            continue
        if not lo <= num <= hi:
            errors[name] = f'Must be between {lo} and {hi}{" " + unit if unit else ""}.'
            continue
        values[name] = num

    return values, errors


def predict(values):
    frame = pd.DataFrame([[values[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    probability = float(MODEL.predict_proba(frame)[0, 1])
    if probability >= 0.7:
        level = 'high'
    elif probability >= 0.4:
        level = 'moderate'
    else:
        level = 'low'

    flags = []
    for name, label, unit, _lo, _hi, normal_lo, normal_hi, _help in FIELDS:
        if normal_lo is None:
            continue
        v = values[name]
        if v < normal_lo or v > normal_hi:
            flags.append({
                'label': label,
                'value': v,
                'unit': unit,
                'range': f'{normal_lo}–{normal_hi}',
                'direction': 'high' if v > normal_hi else 'low',
            })
    return {'probability': round(probability, 3), 'risk': level, 'out_of_range': flags}


@app.route('/')
def home():
    return render_template('index.html', metrics=METRICS)


@app.route('/predict', methods=['GET'])
def form():
    return render_template('predict.html', fields=FIELDS, values={}, errors={})


@app.route('/data_predict', methods=['POST'])  # old URL, kept for compatibility
@app.route('/predict', methods=['POST'])
def submit():
    values, errors = parse_input(request.form)
    if errors:
        return render_template('predict.html', fields=FIELDS, values=request.form, errors=errors), 400
    result = predict(values)
    return render_template('result.html', result=result, metrics=METRICS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error='Send a JSON object with the patient values.'), 400
    values, errors = parse_input(payload)
    if errors:
        return jsonify(errors=errors), 400
    return jsonify(predict(values))


@app.errorhandler(404)
def not_found(_e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=int(os.environ.get('PORT', 5000)))
