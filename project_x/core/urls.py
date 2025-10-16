# core/urls.py (updated excerpt)
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login_view'),
    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('profile/update/', views.user_profile_update, name='user_profile_update'),
    path('loan/apply/', views.loan_application_create, name='loan_application_create'),
    path('loan/status/', views.loan_status, name='loan_status'),
    path('loan/repay/<int:loan_id>/', views.loan_repayment, name='loan_repayment'),
    path('transaction/add/', views.add_transaction, name='add_transaction'),
    path('transactions/all/', views.all_transactions, name='all_transactions'),  # New URL
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('retrain-model/', views.retrain_model, name='retrain_model'),
    path('logout/', views.logout_view, name='logout'),
]