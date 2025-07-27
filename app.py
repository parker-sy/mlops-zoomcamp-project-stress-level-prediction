"""
Flask API for predicting stress levels.
"""

import mlflow
import pickle
import numpy as np
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

app = Flask(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "stress-level-xgb"
MODEL_VERSION = 'Production'


def load_model():
    """Get the production model"""
    try:
        prod_model = mlflow.xgboost.load_model(
            model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}",
        )
        print("Mlflow Model Loaded.")
    except:
        with open("Models/xgb_classifier.bin", "rb") as model:
            prod_model = pickle.load(model)
            print("Local Model Loaded.")
    return prod_model


def predict_stress_level(input_array):
    """Make prediction"""
    model = load_model()
    predicted_level = model.predict(input_array)
    return predicted_level.tolist()[0]


@app.route("/predict", methods=["POST"])
def predict():
    """Parse input and make predcition"""
    data = request.json

    # from [1, 2,...] to [[1, 2, ...]]
    feature = np.array(data["features"]).reshape(1, -1)

    predicted_level = predict_stress_level(feature)

    result = {"prediction": predicted_level}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5020)
