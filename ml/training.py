import os
import pickle
import pandas as pd
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer

from ml.preprocessing import analyze_categorical_columns, decide_encoding
from ml import evaluation


def train_models(csv_path, target):
    # load dataset
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]

    # task detection
    task_type = "classification" if y.nunique() <= 10 else "regression"

    # encoding strategy
    cat_info = analyze_categorical_columns(X)
    onehot_cols, label_cols, drop_cols = decide_encoding(cat_info)
    X = X.drop(columns=drop_cols)
    num_cols = X.select_dtypes(exclude="object").columns

    # transformers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    onehot_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    label_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("onehot", onehot_transformer, onehot_cols),
            ("label", label_transformer, label_cols),
        ],
        remainder="drop"
    )

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model dictionaries
    modelClassifier = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier())
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier())
        ]),
        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier())
        ]),
        "SVC": Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVC())
        ])
    }
    modelRegressor = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "Decision Tree Regressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor())
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor())
        ])
    }

    results = {}
    trained_models = {}

    if task_type == "classification":
        for name, model in modelClassifier.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = round(accuracy_score(y_test, preds), 4)
            trained_models[name] = model
    else:
        for name, model in modelRegressor.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = round(r2_score(y_test, preds), 4)
            trained_models[name] = model

    # ensure directories
    os.makedirs("static/plots", exist_ok=True)
    os.makedirs("media/models", exist_ok=True)

    comparison_plot = evaluation.plot_model_comparison(results, task_type, "static")
    cm_plot = None

    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    if task_type == "classification":
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        cm_plot = evaluation.plot_confusion(cm, "static")

    # feature importances
    feature_importance_plot = None
    final_model = best_model.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
        preprocessor_instance = best_model.named_steps["preprocessor"]

        feature_importance_map = {}
        feature_idx = 0
        for col in num_cols:
            feature_importance_map[col] = importances[feature_idx]
            feature_idx += 1
        if "onehot" in preprocessor_instance.named_transformers_:
            ohe = preprocessor_instance.named_transformers_["onehot"]
            if ohe != "drop" and len(onehot_cols) > 0:
                ohe_feature_names = ohe.get_feature_names_out(onehot_cols)
                for fname in ohe_feature_names:
                    original_col = fname.split("_")[0]
                    feature_importance_map.setdefault(original_col, 0.0)
                    feature_importance_map[original_col] += importances[feature_idx]
                    feature_idx += 1
        if "label" in preprocessor_instance.named_transformers_:
            if len(label_cols) > 0:
                for col in label_cols:
                    feature_importance_map[col] = importances[feature_idx]
                    feature_idx += 1
        feature_importance_plot = evaluation.plot_feature_importances(
            feature_importance_map, "static"
        )

    # versioning
    meta_path = "media/models/best_model_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        version = meta.get("version", 0) + 1
    else:
        version = 1

    versioned_path = f"models/best_model_v{version}.pkl"
    versioned_full = os.path.join("media", versioned_path)
    with open(versioned_full, "wb") as f:
        pickle.dump(best_model, f)
    unversioned_full = os.path.join("media", "models", "best_model.pkl")
    with open(unversioned_full, "wb") as f:
        pickle.dump(best_model, f)

    # metadata
    feature_types = {col: str(X[col].dtype) for col in X.columns}
    feature_categories = {}
    for col in X.columns:
        if X[col].dtype == "object":
            feature_categories[col] = sorted(
                pd.Series(X[col].dropna().unique()).astype(str).tolist()
            )

    model_metadata = {
        "version": version,
        "model_name": best_model_name,
        "task_type": task_type,
        "metric": max(results.values()),
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(X.columns),
        "feature_types": feature_types,
        "feature_categories": feature_categories
    }
    with open(meta_path, "w") as f:
        json.dump(model_metadata, f, indent=4)

    return {
        "task_type": task_type,
        "results": results,
        "best_model": best_model_name,
        "comparison_plot": comparison_plot,
        "cm_plot": cm_plot,
        "feature_importance_plot": feature_importance_plot,
        "model_path": "models/best_model.pkl",
        "model_meta": model_metadata,
    }
