import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMPruningCallback
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, accuracy_score, classification_report, average_precision_score
)

def tfidf_vectorizer(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, labels, vectorizer

def objective_apc(trial, X_train_ml, y_train, X_val_ml, y_val):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    params = {
        'objective': 'binary',
        'metric': 'None',  # We will calculate metrics manually
        'boosting_type': boosting_type,
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'is_unbalance': True  # Automatically adjusts class weights
    }

    # Add bagging parameters only if boosting_type is not GOSS
    if boosting_type != 'goss':
        params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 1.0)
        params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 10)

    # Train LightGBM
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(
        X_train_ml, y_train, 
        eval_set=[(X_val_ml, y_val)], 
        eval_metric='average_precision',  
        early_stopping_rounds=10,
        callbacks=[LightGBMPruningCallback(trial, 'average_precision')],
        verbose=False
    )

    # Predict probabilities and calculate Average Precision Score (AP)
    val_preds_proba = model.predict_proba(X_val_ml)[:, 1]
    apc = average_precision_score(y_val, val_preds_proba)
    return apc  


def calculate_metrics(y_test, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Recall
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall (Sensitivity)": sensitivity,
        "Specificity (TNR)": specificity,
        "ROC_AUC_Score": roc_auc,
        "PR_AUC_Score": pr_auc,
        "F1_Score": f1_score(y_test, y_pred),
        "F2_Score": (5 * precision_score(y_test, y_pred) * sensitivity) / (4 * precision_score(y_test, y_pred) + sensitivity) if (precision_score(y_test, y_pred) + sensitivity) > 0 else 0,
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return pd.DataFrame(metrics)