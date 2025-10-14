from django.contrib import admin
from .models import UserProfile, LoanApplication, Transaction, MLModelPerformance

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'location', 'monthly_income', 'existing_debt', 'created_at']
    list_filter = ['employment_status', 'created_at']
    search_fields = ['user__username', 'location']
    readonly_fields = ['created_at']

@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display = ['id', 'user_profile', 'amount', 'credit_score', 'status', 'submitted_at', 'approved_at']
    list_filter = ['status', 'submitted_at']
    search_fields = ['user_profile__user__username', 'purpose']
    list_editable = ['status', 'credit_score']
    readonly_fields = ['submitted_at', 'approved_at']

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'amount', 'transaction_type', 'transaction_date']
    list_filter = ['transaction_type', 'transaction_date']
    search_fields = ['user_profile__user__username', 'description']

@admin.register(MLModelPerformance)
class MLModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['accuracy', 'last_trained', 'notes']
    list_filter = ['last_trained']
    readonly_fields = ['accuracy', 'score_distribution', 'last_trained']