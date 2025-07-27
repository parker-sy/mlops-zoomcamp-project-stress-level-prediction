"""
Pipeline for model training
"""

import mlflow
import pandas as pd
from prefect import flow, task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

TEST_SIZE = 0.2
DATA_PATH = "Data/SaYoPillow.csv"


def load_data(data_path):
    """Load training data"""
    # loading the data from file
    df = pd.read_csv(data_path)

    col_mapping = {"sr" : "snoring_rate",
                "rr" : "respiration_rate",
                "t" : "body_temperature",
                "lm" : "limb_movement",
                "bo" : "blood_oxygen",
                "rem" : "eye_movement",
                "sr.1" : "sleeping_rate",
                "hr" : "heart_rate",
                "sl" : "stress_level"
                }

    df.rename(columns=col_mapping, inplace=True)

    x = df.drop("stress_level", axis=1)
    y = df["stress_level"]
    # separation of the data on train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=70, random_state=111
    )

    return x_train, x_test, y_train, y_test


@task(log_prints=True, name="Train the model")
def run_model_train():
    """Train and track model"""

    # disabling autologging
    mlflow.xgboost.autolog()
    with mlflow.start_run():

        # loading of train and test datasets
        x_train, x_test, y_train, y_test = load_data(DATA_PATH)
        print("Data Load.")

        # fit model on training data
        model = XGBClassifier()
        print("Training model...")
        model.fit(x_train, y_train)

        # make predictions for test data
        y_pred_train = model.predict(x_train)
        # evaluate predictions
        acc_train = accuracy_score(y_train, y_pred_train)

        # make predictions for test data
        print("Evaluating...")
        y_pred_test = model.predict(x_test)
        # evaluate predictions
        acc_test = accuracy_score(y_test, y_pred_test)

        # logging metrics to mlflow
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)

        # logging parameters to mlflow
        mlflow.log_params(model.get_xgb_params())

        # logging model to mlflow and register the model
        MODEL_NAME = "stress-level-xgboost"
        mlflow.xgboost.log_model(model, name="model")
        print("Model Logged and registered.")


@flow
def main_flow() -> None:
    """The main training pipeline"""

    # setting up mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("stress-level-prediction-xgb")

    # Train
    run_model_train()


if __name__ == "__main__":
    main_flow()
