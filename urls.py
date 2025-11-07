from django.urls import path
from . import views # type: ignore

urlpatterns = [
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_dataset, name='upload'),
    path('analyze/', views.analyze_data, name='analyze'),
    path('cluster/', views.cluster_prediction, name='cluster'),
    path('future/', views.future_prediction, name='future'),
    path('mse/', views.mse_graph, name='mse'),
]
