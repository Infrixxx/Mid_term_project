"""
Test client for Travel Insurance Prediction Web Service.
Sends HTTP requests to test health check and prediction endpoints.
"""

import os
import requests

SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1:9696')

print(f'Testing server at {SERVER_URL}')

# 1. Test Health Endpoint
health_url = f'{SERVER_URL}/health'
try:
    health_resp = requests.get(health_url, timeout=5)
    print('Health Check Status Code:', health_resp.status_code)
    print('Health Check Response:', health_resp.json())
except Exception as e:
    print(f'Health check request failed: {e}')

# 2. Test Customer 1 (High Propensity Profile)
customer_high = {
    'age': 34,
    'employment_type': 'private_sector_self_employed',
    'graduate_or_not': 'yes',
    'annual_income': 1400000,
    'family_members': 6,
    'chronic_diseases': 0,
    'frequent_flyer': 'yes',
    'ever_travelled_abroad': 'yes'
}

predict_url = f'{SERVER_URL}/predict'
print('\\nSending Customer 1 (High Propensity Profile):')
try:
    resp1 = requests.post(predict_url, json=customer_high, timeout=5)
    print('Response Code:', resp1.status_code)
    print('Response Body:', resp1.json())
except Exception as e:
    print(f'Prediction request 1 failed: {e}')

# 3. Test Customer 2 (Low Propensity Profile)
customer_low = {
    'age': 28,
    'employment_type': 'government_sector',
    'graduate_or_not': 'yes',
    'annual_income': 400000,
    'family_members': 3,
    'chronic_diseases': 0,
    'frequent_flyer': 'no',
    'ever_travelled_abroad': 'no'
}

print('\\nSending Customer 2 (Low Propensity Profile):')
try:
    resp2 = requests.post(predict_url, json=customer_low, timeout=5)
    print('Response Code:', resp2.status_code)
    print('Response Body:', resp2.json())
except Exception as e:
    print(f'Prediction request 2 failed: {e}')
