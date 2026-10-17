import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

client = MlflowClient()
model = joblib.load("work/model.joblib")

sample = pd.DataFrame([{
    "distance_km": 7,
    "prep_time_min": 25,
    "traffic_level": 3,
    "rain": 0
}])

with mlflow.start_run(run_name="delivery-model-registration") as run:
    mlflow.log_param("model_type", type(model).__name__)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="delivery_model",
        input_example=sample,
        registered_model_name="delivery-time-model"
    )

    print("RUN_ID:", run.info.run_id)

print("\nREGISTERED MODELS:")
for item in client.search_registered_models():
    print(
        item.name,
        [(v.version, v.current_stage) for v in item.latest_versions]
    )
