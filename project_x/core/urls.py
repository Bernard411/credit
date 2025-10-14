from django.urls import path
from . import views

urlpatterns = [
    path('profile/update/', views.user_profile_update, name='user_profile_update'),
    path('loan/apply/', views.loan_application_create, name='loan_application_create'),
    path('loan/status/', views.loan_status, name='loan_status'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('retrain-model/', views.retrain_model, name='retrain_model'),
]