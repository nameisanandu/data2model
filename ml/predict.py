import os
import pickle
import pandas as pd
from django.conf import settings


def load_model():
    path = os.path.join(settings.MEDIA_ROOT, "models", "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model found")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_dataframe(model, df):
    """Run prediction pipeline on a dataframe and return annotated df."""
    preds = model.predict(df)
    df2 = df.copy()
    df2["Prediction"] = preds
    return df2


def validate_columns(df):
    """Check that df contains the same columns as the training metadata."""
    meta_path = os.path.join(settings.MEDIA_ROOT, "models", "best_model_meta.json")
    if not os.path.exists(meta_path):
        return []
    import json
    with open(meta_path, "r") as f:
        meta = json.load(f)
    required = set(meta.get("features", []))
    missing = required - set(df.columns)
    return sorted(missing)
