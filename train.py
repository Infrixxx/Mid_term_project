"""
Travel Insurance Purchase Prediction Training Script
Trains the final tuned XGBoost classification model on the full training set
and exports the DictVectorizer and model to model.bin.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import xgboost as xgb

# Model configuration and tuned hyperparameters
DATA_PATH = os.path.join('data', 'travel_insurance.csv')
MODEL_OUTPUT_PATH = 'model.bin'
DECISION_THRESHOLD = 0.40

CATEGORICAL_COLUMNS = [
    'employment_type',
    'graduate_or_not',
    'frequent_flyer',
    'ever_travelled_abroad'
]

NUMERICAL_COLUMNS = [
    'age',
    'annual_income',
    'family_members',
    'chronic_diseases'
]

TARGET_COLUMN = 'travel_insurance'

XGB_PARAMS = {
    'eta': 0.03,
    'max_depth': 4,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}

NUM_BOOST_ROUNDS = 80


def load_and_prepare_data(file_path):
    print(f'Loading data from {file_path}')
    df = pd.read_csv(file_path)
    print(f'Dataset shape: {df.shape}')
    return df


def split_data(df):
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )

    y_full_train = df_full_train[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values

    df_full_train = df_full_train.drop(columns=[TARGET_COLUMN])
    df_test = df_test.drop(columns=[TARGET_COLUMN])

    print(f'Full training records: {len(df_full_train)}')
    print(f'Held out test records: {len(df_test)}')
    return df_full_train, df_test, y_full_train, y_test


def vectorize_features(df_train, df_test):
    feature_cols = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    train_dicts = df_train[feature_cols].to_dict(orient='records')
    test_dicts = df_test[feature_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)
    return dv, X_train, X_test


def train_model(X_train, y_train, feature_names):
    print('Training tuned XGBoost model')
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUNDS)
    return model


def evaluate_model(model, dv, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(dv.get_feature_names_out()))
    y_pred_proba = model.predict(dtest)
    y_pred_binary = (y_pred_proba >= DECISION_THRESHOLD).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print('Held Out Test Set Evaluation Results:')
    print(f' ROC AUC Score:    {auc:.4f}')
    print(f' Accuracy:         {acc:.4f}')
    print(f' Precision:        {prec:.4f}')
    print(f' Recall:           {rec:.4f}')
    print(f' F1 Score:         {f1:.4f}')
    print('\\nDetailed Classification Report:')
    print(classification_report(y_test, y_pred_binary))


def save_artifacts(dv, model, threshold, output_path):
    print(f'Saving artifacts to {output_path}')
    with open(output_path, 'wb') as f_out:
        pickle.dump((dv, model, threshold), f_out)
    print('Model artifacts successfully saved')


def main():
    df = load_and_prepare_data(DATA_PATH)
    df_full_train, df_test, y_full_train, y_test = split_data(df)
    dv, X_full_train, X_test = vectorize_features(df_full_train, df_test)
    model = train_model(X_full_train, y_full_train, dv.get_feature_names_out())
    evaluate_model(model, dv, X_test, y_test)
    save_artifacts(dv, model, DECISION_THRESHOLD, MODEL_OUTPUT_PATH)
    print('Training pipeline complete')


if __name__ == '__main__':
    main()
