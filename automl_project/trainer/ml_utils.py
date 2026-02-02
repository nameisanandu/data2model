# ================================
# ML Utilities for AutoML Web App
# ================================

import os
import pickle
import pandas as pd
import json
from datetime import datetime


# ---- Matplotlib (NON-GUI backend for Django) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Scikit-learn imports ----
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer



# ================================
# Auto-detect encoding per column
# ================================

def analyze_categorical_columns(X):
    """
    Returns dict: {column_name: unique_count}
    """
    cat_info = {}
    for col in X.select_dtypes(include="object").columns:
        cat_info[col] = X[col].nunique()
    return cat_info


def decide_encoding(cat_info):
    """
    Decide encoding strategy based on cardinality
    """
    onehot_cols = []
    label_cols = []
    drop_cols = []

    for col, unique_count in cat_info.items():
        if unique_count <= 2:
            label_cols.append(col)
        elif unique_count <= 10:
            onehot_cols.append(col)
        elif unique_count <= 50:
            onehot_cols.append(col)   # safe default
        else:
            drop_cols.append(col)     # high-cardinality noise

    return onehot_cols, label_cols, drop_cols








# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================
def train_models(csv_path, target):

    # --------------------
    # Load dataset
    # --------------------
    df = pd.read_csv(csv_path)
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    # --------------------
    # Task detection
    # --------------------
    if y.nunique() <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

  


    # ----------------------------------
    # Auto-detect encoding strategy
    # ----------------------------------
    cat_info = analyze_categorical_columns(X)
    onehot_cols, label_cols, drop_cols = decide_encoding(cat_info)

    # Drop high-cardinality columns
    X = X.drop(columns=drop_cols)

    num_cols = X.select_dtypes(exclude="object").columns

    # ----------------------------------
    # Transformers
    # ----------------------------------
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

    # ----------------------------------
    # ColumnTransformer (AutoML-style)
    # ----------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("onehot", onehot_transformer, onehot_cols),
            ("label", label_transformer, label_cols),
        ],
        remainder="drop"
    )





    # --------------------
    # Train-test split
    # --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------
    # Model dictionaries
    # --------------------
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

    # --------------------
    # Train models
    # --------------------
    if task_type == "classification":
        for name, model in modelClassifier.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            results[name] = round(score, 4)
            trained_models[name] = model

    else:
        for name, model in modelRegressor.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            results[name] = round(score, 4)
            trained_models[name] = model

    # --------------------
    # Create static folders
    # --------------------
    os.makedirs("static/plots", exist_ok=True)
    os.makedirs("media/models", exist_ok=True)

    # --------------------
    # Model comparison plot
    # --------------------
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title(f"{task_type.upper()} Model Comparison")
    plt.ylabel("Accuracy" if task_type == "classification" else "R2 Score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    comparison_plot = "plots/model_comparison.png"
    plt.savefig(f"static/{comparison_plot}")
    plt.close()

    # --------------------
    # Select best model
    # --------------------
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    # --------------------
    # Confusion Matrix (Classification only)
    # --------------------
    cm_plot = None
    if task_type == "classification":
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_plot = "plots/confusion_matrix.png"
        plt.savefig(f"static/{cm_plot}")
        plt.close()

    feature_importance_plot = None

    final_model = best_model.named_steps["model"]

    if hasattr(final_model, "feature_importances_"):

        importances = final_model.feature_importances_
        preprocessor = best_model.named_steps["preprocessor"]

        feature_importance_map = {}
        feature_idx = 0  # pointer over transformed features

        # --------------------
        # Numeric features
        # --------------------
        for col in num_cols:
            feature_importance_map[col] = importances[feature_idx]
            feature_idx += 1

        # --------------------
        # One-Hot Encoded features
        # --------------------
        if "onehot" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["onehot"]

            if ohe != "drop" and len(onehot_cols) > 0:
                ohe_feature_names = ohe.get_feature_names_out(onehot_cols)

                for fname in ohe_feature_names:
                    original_col = fname.split("_")[0]
                    feature_importance_map.setdefault(original_col, 0.0)
                    feature_importance_map[original_col] += importances[feature_idx]
                    feature_idx += 1

        # --------------------
        # Ordinal / Label Encoded features
        # --------------------
        if "label" in preprocessor.named_transformers_:
            if len(label_cols) > 0:
                for col in label_cols:
                    feature_importance_map[col] = importances[feature_idx]
                    feature_idx += 1

        # --------------------
        # Convert to DataFrame
        # --------------------
        fi_df = pd.DataFrame(
            feature_importance_map.items(),
            columns=["Feature", "Importance"]
        )

        # --------------------
        # Top N columns only
        # --------------------
        TOP_N = 10
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(TOP_N)

        # --------------------
        # Plot
        # --------------------
        plt.figure(figsize=(6, 4))
        plt.barh(fi_df["Feature"], fi_df["Importance"])
        plt.xlabel("Importance")
        plt.title("Top Feature Importances (Column Level)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        feature_importance_plot = "plots/feature_importance.png"
        plt.savefig(f"static/{feature_importance_plot}")
        plt.close()

    # --------------------
    # Save trained model
    # --------------------


# --------------------
# Model Versioning
# --------------------
    os.makedirs("media/models", exist_ok=True)

    meta_path = "media/models/best_model_meta.json"

    # Load previous version
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        version = meta.get("version", 0) + 1
    else:
        version = 1

    model_filename = f"best_model_v{version}.pkl"
    model_path = f"models/{model_filename}"
    full_model_path = os.path.join("media", model_path)

    with open(full_model_path, "wb") as f:
        pickle.dump(best_model, f)




# Save model metadata
# --------------------
    model_metadata = {
        "version": version,
        "model_name": best_model_name,
        "task_type": task_type,
        "metric": max(results.values()),
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(X.columns)
    }

    with open(meta_path, "w") as f:
        json.dump(model_metadata, f, indent=4)


    # --------------------
    # Return results
    # --------------------





    return {
        "task_type": task_type,
        "results": results,
        "best_model": best_model_name,
        "comparison_plot": comparison_plot,
        "cm_plot": cm_plot,
        "feature_importance_plot": feature_importance_plot,
        "model_path": model_path,
        "model_meta": model_metadata,
        "model_path": "models/best_model.pkl"
    }
