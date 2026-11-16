# Travel Insurance Purchase Propensity Prediction

Machine Learning Zoomcamp Midterm Project

## 1. Project Overview and Business Problem

Travel platforms and digital agencies experience massive search volume, but ancillary revenue products such as travel insurance often suffer from poor conversion rates. Indiscriminately displaying insurance popups to every visitor damages user experience and causes digital fatigue. Conversely, ignoring high propensity travelers results in lost gross merchandise value.

This project delivers a complete machine learning system that predicts whether a traveler will purchase travel insurance. By evaluating demographic indicators, household size, employment category, and travel history, the service computes a purchase propensity score in real time.

### Real World Solution Workflow

1. A customer initiates flight or hotel reservations on a digital travel portal.
2. The booking portal gathers user profile features and transmits a JSON payload to the prediction web service.
3. The machine learning model computes the purchase probability and evaluates it against an optimized decision threshold.
4. If purchase probability meets or exceeds the threshold, the portal displays personalized insurance options with tailored benefits.
5. If the traveler exhibits low propensity, the portal provides a streamlined checkout without promotional interruptions.

## 2. Dataset Information

The project utilizes the Travel Insurance Prediction dataset.
Location: data/travel_insurance.csv

The dataset contains 1987 traveler records across 9 standardized attributes.

### Features

1. age: Age of customer between 25 and 35 years old (Numerical)
2. employment_type: Employment sector, categorized as private sector self employed or government sector (Categorical)
3. graduate_or_not: Indicator of university degree completion (Categorical: yes or no)
4. annual_income: Annual earnings in local currency units (Numerical)
5. family_members: Number of individuals in traveler household from 1 to 9 (Numerical)
6. chronic_diseases: Indicator of chronic health condition presence (Numerical binary: 0 or 1)
7. frequent_flyer: History of frequent airline ticket purchases (Categorical: yes or no)
8. ever_travelled_abroad: Record of past international travel history (Categorical: yes or no)
9. travel_insurance: Target variable indicating whether customer purchased insurance (Binary: 0 = No, 1 = Yes)

Class Distribution:
Did not purchase (0): 1277 records (64.3 percent)
Purchased (1): 710 records (35.7 percent)

## 3. Project Structure

Repository files and functions:

1. data/travel_insurance.csv: Cleaned traveler dataset.
2. notebook.ipynb: Executed Jupyter notebook containing complete data cleaning, extensive exploratory data analysis, feature importance calculations, multiple model training, hyperparameter tuning, model comparison, threshold analysis, test evaluation, and model export.
3. train.py: Python script to train the final tuned model on full training data and export artifacts to model.bin.
4. predict.py: Production web service built with Flask serving predictions over HTTP POST endpoints.
5. predict_test.py: Test script validating the web service with both high and low propensity customer profiles.
6. model.bin: Serialized model binary containing DictVectorizer, tuned XGBoost model, and optimal decision threshold.
7. requirements.txt: Python dependencies and versions.
8. Pipfile: Dependency specification for Pipenv users.
9. Dockerfile: Container definition for packaging and deploying the prediction service.
10. .dockerignore: Ignore rules for lightweight container builds.
11. deployment.yaml: Kubernetes Deployment manifest with readiness and liveness probes.
12. service.yaml: Kubernetes Service manifest for traffic routing.
13. README.md: Comprehensive documentation.

## 4. Exploratory Data Analysis and Key Insights

The exploratory data analysis in notebook.ipynb produced several decisive business findings:

1. International Travel Experience: Travelers who have previously traveled abroad demonstrate a 78.4 percent purchase rate compared to 25.6 percent for travelers without foreign travel history. This attribute had the highest Mutual Information score (0.1384) with the target.
2. Annual Income: Customers earning above 1,300,000 currency units exhibit an insurance purchase rate exceeding 85 percent, whereas lower income tiers stay below 20 percent.
3. Frequent Flyer Status: Frequent flyers show a 57.3 percent purchase rate, whereas infrequent flyers purchase at 30.0 percent.
4. Family Size: Families with more than 4 members show elevated insurance interest due to higher aggregate risk exposure per trip.
5. Age and Employment: Private sector and self employed professionals purchase insurance at 40.2 percent compared to 24.6 percent for government employees.

## 5. Model Training and Comparison

The dataset was split using stratified partitioning:
Training set: 60 percent (1192 samples)
Validation set: 20 percent (397 samples)
Test set: 20 percent (398 samples)

Features were transformed using DictVectorizer. Five distinct model architectures were trained and systematically tuned on the validation set.

### Validation Results Summary

1. XGBoost:
ROC AUC: 0.7753
Accuracy: 0.8161
Precision: 0.8875
Recall: 0.5000
F1 Score: 0.6396
Tuned Parameters: max_depth=4, min_child_weight=3, eta=0.03, num_boost_round=80

2. Decision Tree:
ROC AUC: 0.7719
Accuracy: 0.8086
Precision: 0.8659
Recall: 0.5000
F1 Score: 0.6339
Tuned Parameters: max_depth=6, min_samples_leaf=5

3. Random Forest:
ROC AUC: 0.7674
Accuracy: 0.8060
Precision: 0.8642
Recall: 0.4930
F1 Score: 0.6278
Tuned Parameters: n_estimators=100, max_depth=4, min_samples_leaf=3

4. LightGBM:
ROC AUC: 0.7652
Accuracy: 0.8136
Precision: 0.8625
Recall: 0.4859
F1 Score: 0.6216
Tuned Parameters: learning_rate=0.03, num_leaves=15, num_boost_round=80

5. Logistic Regression:
ROC AUC: 0.6858
Accuracy: 0.7481
Precision: 0.7818
Recall: 0.3028
F1 Score: 0.4365
Tuned Parameters: solver=liblinear, C=1.0

### Final Champion Model Selection

XGBoost achieved the highest validation ROC AUC (0.7753). Tree based models significantly outperformed linear regression due to strong non linear interactions between income thresholds and international travel habits.

### Final Evaluation on Held Out Test Set

Training the final tuned XGBoost model on the combined training and validation set and evaluating on the untouched test set yielded:
Test ROC AUC: 0.7927
Test Accuracy: 0.8141
Test Precision: 0.8953
Test Recall: 0.5423
Test F1 Score: 0.6754

The test score confirms strong generalization without overfitting.

## 6. Local Setup and Reproduction Guide

### Prerequisites

Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your workstation.
Git installed.

### Step 1: Clone Repository

git clone https://github.com/Infrixxx/Mid_term_project.git
cd Mid_term_project

### Step 2: Virtual Environment Setup

Option A: Standard Virtualenv

python -m venv venv

On Linux or macOS:
source venv/bin/activate

On Windows:
.\\venv\\Scripts\\activate

pip install --upgrade pip
pip install -r requirements.txt

Option B: Pipenv

pip install pipenv
pipenv install
pipenv shell

### Step 3: Run Exploratory Data Analysis Notebook

To view or rerun the notebook:

jupyter notebook notebook.ipynb

You can also execute the notebook from the terminal:

python -m nbclient notebook.ipynb

### Step 4: Run Training Script

The training script trains the champion XGBoost model on full training data and updates model.bin:

python train.py

### Step 5: Start Prediction Web Service

Run the Flask web service:

python predict.py

The service listens on http://127.0.0.1:9696.

### Step 6: Test Local Web Service

In a separate terminal window, run the automated test client:

python predict_test.py

Sample expected terminal output:

Testing server at http://127.0.0.1:9696
Health Check Status Code: 200
Health Check Response: {'decision_threshold': 0.4, 'model_type': 'xgboost', 'service': 'travel_insurance_prediction', 'status': 'healthy'}

Sending Customer 1 (High Propensity Profile):
Response Code: 200
Response Body: {'decision_threshold': 0.4, 'purchase_probability': 0.8822, 'recommendation': 'Offer travel insurance package during checkout', 'target_insurance': True}

Sending Customer 2 (Low Propensity Profile):
Response Code: 200
Response Body: {'decision_threshold': 0.4, 'purchase_probability': 0.1162, 'recommendation': 'Standard checkout without insurance prompt', 'target_insurance': False}

You can also send a request using curl:

curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\": 34, \"employment_type\": \"private_sector_self_employed\", \"graduate_or_not\": \"yes\", \"annual_income\": 1400000, \"family_members\": 6, \"chronic_diseases\": 0, \"frequent_flyer\": \"yes\", \"ever_travelled_abroad\": \"yes\"}"

## 7. Containerization with Docker

The application is containerized with a production multi worker WSGI server.

### Build Docker Image

docker build -t travel_insurance_prediction:latest .

### Run Docker Container

docker run -it --rm -p 9696:9696 travel_insurance_prediction:latest

The container launches 4 Gunicorn worker processes listening on port 9696 with automated health check monitoring.

### Test Containerized Service

Run the test client against the running container:

python predict_test.py

## 8. Kubernetes and Cloud Deployment

### Kubernetes Local or Remote Cluster Deployment

The repository includes complete Kubernetes manifests:
1. deployment.yaml: Configures 2 replicas, container resource requests, memory limits, and HTTP health check probes.
2. service.yaml: Exposes the deployment as a LoadBalancer service.

To deploy on Minikube, Kind, or a managed cloud Kubernetes cluster:

1. Start your local cluster:
minikube start

2. Load the local Docker image into Minikube:
minikube image load travel_insurance_prediction:latest

3. Apply manifests:
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

4. Verify cluster resources:
kubectl get pods
kubectl get deployments
kubectl get services

5. Expose service port for local verification:
kubectl port-forward service/travel-insurance-prediction-service 9696:80

6. Test the Kubernetes deployment:
python predict_test.py

### Cloud Deployment to Render, Fly.io, AWS, or GCP

Option A: Render Web Service

1. Fork or push this repository to GitHub.
2. Create an account on render.com.
3. Click New Web Service and connect the repository.
4. Select Docker as the Runtime environment.
5. Set the instance port environment variable PORT to 9696.
6. Click Deploy Web Service. Render builds the Docker image and publishes a public URL.

Option B: Fly.io

1. Install Fly CLI: flyctl
2. Authenticate: fly auth login
3. Launch app: fly launch --name travel-insurance-prediction
4. Deploy container: fly deploy
5. Test public endpoint:
curl https://travel-insurance-prediction.fly.dev/health

Option C: AWS Elastic Beanstalk or AWS ECS Fargate

1. Build and tag the image for Amazon ECR:
docker tag travel_insurance_prediction:latest <account_id>.dkr.ecr.<region>.amazonaws.com/travel_insurance_prediction:latest
2. Authenticate and push:
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/travel_insurance_prediction:latest
3. Deploy ECS task definition referencing port 9696 with Application Load Balancer.

Option D: Google Cloud Run

1. Build and submit image to Google Artifact Registry:
gcloud builds submit --tag gcr.io/<project_id>/travel_insurance_prediction:latest
2. Deploy to Cloud Run:
gcloud run deploy travel-insurance-prediction --image gcr.io/<project_id>/travel_insurance_prediction:latest --platform managed --port 9696 --allow-unauthenticated
