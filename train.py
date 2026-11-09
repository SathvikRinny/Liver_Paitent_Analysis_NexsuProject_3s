"""Train the liver disease model.

Usage:
    python train.py

Reads data/indian_liver_patient.csv, compares a few models with stratified
5-fold cross-validation, keeps the best one by ROC AUC, reports its scores on
a held-out test set, and saves it to model/liver_model.joblib together with
model/metrics.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / 'data' / 'indian_liver_patient.csv'
MODEL_DIR = ROOT / 'model'
SEED = 42

FEATURES = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio',
]
# Lab values are heavily right-skewed, so they are log-transformed before scaling.
SKEWED = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
          'Alamine_Aminotransferase', 'Aspartate_Aminotransferase']


def load_data():
    df = pd.read_csv(DATA)
    df['Gender'] = (df['Gender'].str.strip().str.lower() == 'male').astype(int)
    X = df[FEATURES]
    y = (df['Dataset'] == 1).astype(int)  # 1 = liver patient, 0 = not
    return X, y


def build_pipeline(model):
    pre = ColumnTransformer([
        ('skewed', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
            ('scale', StandardScaler()),
        ]), SKEWED),
        ('other', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), [f for f in FEATURES if f not in SKEWED]),
    ])
    return Pipeline([('pre', pre), ('model', model)])


CANDIDATES = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                            class_weight='balanced', random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(random_state=SEED),
}


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train, cv=cv, scoring='roc_auc')
        cv_scores[name] = round(float(scores.mean()), 3)
        print(f'{name:<20} CV ROC AUC = {scores.mean():.3f} ± {scores.std():.3f}')

    best_name = max(cv_scores, key=cv_scores.get)
    pipe = build_pipeline(CANDIDATES[best_name]).fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test = {
        'roc_auc': round(roc_auc_score(y_test, proba), 3),
        'accuracy': round(accuracy_score(y_test, pred), 3),
        'precision': round(precision_score(y_test, pred), 3),
        'recall': round(recall_score(y_test, pred), 3),
        'f1': round(f1_score(y_test, pred), 3),
    }
    print(f'\nBest model: {best_name}')
    print('Test set:', test)

    # Refit on all data for the deployed model.
    final = build_pipeline(CANDIDATES[best_name]).fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / 'liver_model.joblib', compress=3)
    metrics = {
        'model': best_name,
        'cv_roc_auc': cv_scores,
        'test': test,
        'n_samples': int(len(X)),
        'features': FEATURES,
        'sklearn_version': sklearn.__version__,
    }
    (MODEL_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Saved model to {MODEL_DIR / "liver_model.joblib"}')


if __name__ == '__main__':
    main()
