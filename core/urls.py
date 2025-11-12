from django.urls import path
from . import views
# Define URL structures for frontend navigation
urlpatterns = [
    path('', views.login_view, name='login'),      # root = login page
    path('home/', views.index, name='home'),       # welcome page
    path('logout/', views.logout_view, name='logout'), # logout page
    path('upload/', views.upload_dataset, name='upload'), # dataset upload page
    path('analyze/', views.analyze_data, name='analyze'), # data analysis page
    path('cluster/', views.cluster_prediction, name='cluster'), # clustering page
    path('future/', views.future_prediction, name='future'), # future prediction page
]
