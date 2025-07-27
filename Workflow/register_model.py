"""
Register model
"""

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("stress-level-prediction-xgb")
client = MlflowClient()


# Pulls the top 3 runs sorted by test accuracy in descending order.
runs = client.search_runs(
    experiment_ids="1",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=3,
    order_by=["metrics.accuracy_test DESC"],
)

# the best run's model URI
run_id = runs[0].info.run_id
MODEL_URL = f"runs:/{run_id}/model"
# artifact_uri = runs[0].info.artifact_uri
# MODEL_URL = f"file:/{artifact_uri}"
MODEL_NAME = "stress-level-xgb"

# Registering the model
result = mlflow.register_model(model_uri=MODEL_URL, name=MODEL_NAME)
print(result)
print(f"Model {MODEL_NAME} has been registered, with run id {run_id}")

# Promoting it to Production
NEW_STAGE = "Production"
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage=NEW_STAGE,
    archive_existing_versions=False,
)

print(f"Model {MODEL_NAME} has been moved to {NEW_STAGE} stage.")
