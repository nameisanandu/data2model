

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd

from .ml_utils import train_models



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
                "fi_plot": output["feature_importance_plot"]
                
,
            }
        )


import os
import pickle
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd

def predict_page(request):
    return render(request, "trainer/predict.html")

def predict_result(request):
    if request.method == "POST":

        import os
        import pickle
        import pandas as pd
        from django.conf import settings
        from django.core.files.storage import FileSystemStorage
        from django.shortcuts import render

        model_path = os.path.join(settings.MEDIA_ROOT, "models", "best_model.pkl")

        # 🔒 SAFETY CHECK
        if not os.path.exists(model_path):
            return render(request, "trainer/predict.html", {
                "error": "No trained model found. Please train a model first."
            })

        # Load trained model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Save uploaded CSV
        csv_file = request.FILES.get("csv_file")
        fs = FileSystemStorage(location="media/uploads/")
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        df = pd.read_csv(file_path)

        # Predict
        predictions = model.predict(df)
        df["Prediction"] = predictions

        # Save predictions
        os.makedirs("media/predictions", exist_ok=True)
        output_path = "media/predictions/predictions.csv"
        df.to_csv(output_path, index=False)

        return render(request, "trainer/predict_result.html", {
            "tables": df.head(10).to_html(classes="table table-bordered"),
            "file_url": settings.MEDIA_URL + "predictions/predictions.csv"
        })
