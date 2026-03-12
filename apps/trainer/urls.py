


from django.urls import path
from .views import upload_csv, train, predict_page, predict_result

urlpatterns = [
    path('', upload_csv, name='upload'),
    path('train/', train, name='train'),
    path('predict/', predict_page, name='predict'),
    path('predict/result/', predict_result, name='predict_result'),
]






