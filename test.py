import mlflow

mlflow.set_tracking_uri("file:artifacts/mlflow")
mlflow.set_experiment("Default")  # experiment名が一致していればOK

with mlflow.start_run(run_name="demo-run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("val_accuracy", 0.92)
