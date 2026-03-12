import pandas as pd


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
