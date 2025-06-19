import pickle
import json
from typing import Dict, List

def save_production_model(model, output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

def save_production_columns(columns: List[str], output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(columns, f)

def save_production_metrics(metrics: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(metrics, f)
