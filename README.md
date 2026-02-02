![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Django](https://img.shields.io/badge/Django-4.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)



# AutoML Django Web Application

An end-to-end AutoML web application built with Django and scikit-learn.
The system automates data preprocessing, model training, evaluation, interpretability, and prediction for tabular datasets, providing a complete machine learning workflow through a web interface.

## Features

* Upload CSV datasets
* Automatic task detection (Classification / Regression)
* Automated preprocessing (scaling, encoding, imputation)
* AutoML model training and comparison
* Best model selection
* Confusion matrix and feature importance visualization
* Column-level interpretability
* Prediction on new datasets
* Download trained ML model (.pkl)
* Model versioning and metadata

## Tech Stack

* Python
* Django
* scikit-learn
* Pandas, NumPy
* Matplotlib

## Project Structure





automl\_web\_app/

│── automl\_project/

│── venv/ (ignored)

│── requirements.txt

│── README.md

│── .gitignore






## Screenshots



### Dataset Upload
![Upload](screenshots/upload.jpeg)



### Target Feature selection
![Target Feature](screenshots/target.jpeg)




### Result
![Result](screenshots/result.jpeg)



### Test Prediction
![Prediction](screenshots/test.jpeg)

### Prediction Preview
![Prediction](screenshots/prediction.jpeg)









How to Run Locally



```bash

git clone https://github.com/YOUR\_USERNAME/automl-django-app.git

cd automl-django-app



python -m venv venv

venv\\Scripts\\activate



pip install -r requirements.txt

cd automl\_project

python manage.py migrate

python manage.py runserver



