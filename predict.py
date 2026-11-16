"""
Travel Insurance Purchase Prediction Web Service
Loads the trained XGBoost model and DictVectorizer from model.bin
and serves predictions over a Flask HTTP API.
"""

import os
import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

MODEL_FILE = os.environ.get('MODEL_FILE', 'model.bin')

print(f'Loading model artifacts from {MODEL_FILE}')
with open(MODEL_FILE, 'rb') as f_in:
    dv, model, threshold = pickle.load(f_in)
print('Model artifacts loaded successfully')

app = Flask('travel_insurance_prediction')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'travel_insurance_prediction',
        'model_type': 'xgboost',
        'decision_threshold': threshold
    })


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    if not customer:
        return jsonify({'error': 'No input data provided'}), 400

    feature_matrix = dv.transform([customer])
    feature_names = list(dv.get_feature_names_out())
    dmatrix = xgb.DMatrix(feature_matrix, feature_names=feature_names)

    probability = float(model.predict(dmatrix)[0])
    target_insurance = bool(probability >= threshold)

    if target_insurance:
        recommendation = 'Offer travel insurance package during checkout'
    else:
        recommendation = 'Standard checkout without insurance prompt'

    response = {
        'purchase_probability': round(probability, 4),
        'target_insurance': target_insurance,
        'decision_threshold': threshold,
        'recommendation': recommendation
    }

    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9696))
    app.run(debug=False, host='0.0.0.0', port=port)
