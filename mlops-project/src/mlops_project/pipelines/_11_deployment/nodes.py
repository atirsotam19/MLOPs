# pipelines/deployment/nodes.py

def save_production_model(model, path):
    import joblib
    joblib.dump(model, path)
    return model  # Return value so Kedro can track it

def save_production_columns(columns, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(columns, f)
    return columns

def save_production_metrics(metrics, path):
    import json
    with open(path, "w") as f:
        json.dump(metrics, f)
    return metrics
