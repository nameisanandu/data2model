import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_model_comparison(results, task_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title(f"{task_type.upper()} Model Comparison")
    plt.ylabel("Accuracy" if task_type == "classification" else "R2 Score")
    plt.xticks(rotation=30)
    plt.tight_layout()

    comparison_plot = "plots/model_comparison.png"
    plt.savefig(os.path.join(output_dir, comparison_plot))
    plt.close()
    return comparison_plot


def plot_confusion(cm, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_plot = "plots/confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, cm_plot))
    plt.close()
    return cm_plot


def plot_feature_importances(feature_importance_map, output_dir, top_n=10):
    os.makedirs(output_dir, exist_ok=True)
    fi_df = pd.DataFrame(
        feature_importance_map.items(),
        columns=["Feature", "Importance"]
    )
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(6, 4))
    plt.barh(fi_df["Feature"], fi_df["Importance"])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (Column Level)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fi_plot = "plots/feature_importance.png"
    plt.savefig(os.path.join(output_dir, fi_plot))
    plt.close()
    return fi_plot
