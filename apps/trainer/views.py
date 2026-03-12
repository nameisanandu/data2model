

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd

from ml.training import train_models
from ml.predict import load_model, predict_dataframe, validate_columns



def upload_csv(request):
    """
    Handles CSV upload and shows target column selection
    """
    if request.method == "POST":
        csv_file = request.FILES.get("csv_file")

        # Save uploaded file
        fs = FileSystemStorage(location="media/uploads/")
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        # Read CSV
        df = pd.read_csv(file_path)

        # Store path in session
        request.session["csv_path"] = file_path

        return render(
            request,
            "trainer/preview.html",
            {
                "columns": df.columns
            }
        )

    return render(request, "trainer/upload.html")




def train(request):
    """
    Triggers AutoML training based on selected target column
    """
    if request.method == "POST":
        target = request.POST.get("target")
        csv_path = request.session.get("csv_path")

        # Run AutoML
        output = train_models(csv_path, target)

        # training finished; remove csv_path from session so the next train
        # requires a fresh upload.  This prevents accidentally re‑training on
        # the previous file when the user forgets to upload a new one.
        try:
            del request.session["csv_path"]
        except KeyError:
            pass

        return render(
            request,
            "trainer/results.html",
            {
                "task_type": output["task_type"],
                "results": output["results"],
                "best_model": output["best_model"],
                "acc_plot": output["comparison_plot"],
                "cm_plot": output["cm_plot"],
                "model_meta": output["model_meta"],
                "model_path": output["model_path"],
                "fi_plot": output["feature_importance_plot"],
                # expose features so template can show them directly
                "trained_features": output["model_meta"].get("features", []),
            }
        )


import os
import pickle
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd

def predict_page(request):
    # Attempt to load metadata so we can inform the user which columns are
    # expected by the current model.  If the metadata file is missing, we'll
    # just render the form without details.
    required_cols = []
    feature_types = {}
    feature_categories = {}
    inputs = []  # list of dicts {col, type, categories}

    meta_path = os.path.join(settings.MEDIA_ROOT, "models", "best_model_meta.json")
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            required_cols = meta.get("features", [])
            feature_types = meta.get("feature_types", {})
            feature_categories = meta.get("feature_categories", {})
        except Exception:
            required_cols = []
            feature_types = {}
            feature_categories = {}

    # build simple input spec list so the template stays clean
    for col in required_cols:
        dtype = feature_types.get(col, "").lower()
        if col in feature_categories and feature_categories[col]:
            # if we have category list, prefer a select element
            inputs.append({
                "col": col,
                "type": "select",
                "categories": feature_categories[col]
            })
        else:
            if "int" in dtype or "float" in dtype:
                typ = "number"
            else:
                typ = "text"
            inputs.append({"col": col, "type": typ, "categories": None})

    return render(request, "trainer/predict.html", {"required_cols": required_cols, "feature_types": feature_types, "inputs": inputs})

def predict_result(request):
    if request.method == "POST":

        import os
        import pickle
        import pandas as pd
        from django.conf import settings
        from django.core.files.storage import FileSystemStorage
        from django.shortcuts import render

        # attempt to load the current model using helper; will raise if missing
        try:
            model = load_model()
        except FileNotFoundError:
            return render(request, "trainer/predict.html", {
                "error": "No trained model found. Please train a model first."
            })

        # determine whether the user supplied a file or manual values
        if "manual" in request.POST:
            # build data frame from POST values using feature list
            # (dtype conversion can be delegated to pandas/sklearn pipeline)
            data = {}
            meta_path = os.path.join(settings.MEDIA_ROOT, "models", "best_model_meta.json")
            if os.path.exists(meta_path):
                try:
                    import json
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    required_cols = meta.get("features", [])
                except Exception:
                    required_cols = []
            else:
                required_cols = []

            for col in required_cols:
                data[col] = request.POST.get(col, "")

            df = pd.DataFrame([data])

            # no need to validate column presence here because we built it
        else:
            # Save uploaded CSV
            csv_file = request.FILES.get("csv_file")
            fs = FileSystemStorage(location="media/uploads/")
            filename = fs.save(csv_file.name, csv_file)
            file_path = fs.path(filename)

            df = pd.read_csv(file_path)

            # --------------------------------------------------
            # Validate uploaded data against training features
            # --------------------------------------------------
            required_cols = None
            meta_path = os.path.join(settings.MEDIA_ROOT, "models", "best_model_meta.json")
            if os.path.exists(meta_path):
                try:
                    import json
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    required_cols = set(meta.get("features", []))
                except Exception:
                    required_cols = None

            if required_cols is not None:
                missing = required_cols - set(df.columns)
                if missing:
                    return render(request, "trainer/predict.html", {
                        "error": (
                            "Uploaded CSV is missing the following columns "
                            f"required by the model: {sorted(missing)}"
                        )
                    })

        # Predict
        # validate columns only when user uploaded CSV
        if "manual" not in request.POST:
            missing = validate_columns(df)
            if missing:
                return render(request, "trainer/predict.html", {
                    "error": (
                        "Uploaded CSV is missing the following columns "
                        f"required by the model: {missing}"
                    )
                })
        try:
            df = predict_dataframe(model, df)
        except ValueError as e:
            return render(request, "trainer/predict.html", {"error": str(e)})

        # Save predictions
        os.makedirs("media/predictions", exist_ok=True)
        output_path = "media/predictions/predictions.csv"
        df.to_csv(output_path, index=False)

        return render(request, "trainer/predict_result.html", {
            "tables": df.head(10).to_html(classes="table table-bordered"),
            "file_url": settings.MEDIA_URL + "predictions/predictions.csv"
        })
