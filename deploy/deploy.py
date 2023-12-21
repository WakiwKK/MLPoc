import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the MLflow model
mlflow.set_tracking_uri("http://airflow-docker-mlflow-server-1:5000/")
mlflow.set_experiment(experiment_id="0")
runs = mlflow.search_runs()

# best_accuracy_model = runs.sort_values(by="matrics.accuracy", ascending =False)
# best_model_name = best_accuracy_model['registered_model_name']
# best_model_version = best_accuracy_model['version']

latest_run = runs.iloc[0]
model_uri = latest_run['artifact_uri'] + '/best_estimator'
loaded_model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame(data['features'])  # Assuming data is a dictionary with a 'features' key
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(features)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)