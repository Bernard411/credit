from django import forms
from .models import UserProfile, LoanApplication

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'gender', 'location', 'employment_status', 'monthly_income', 'monthly_expenses', 'existing_debt', 'number_of_dependents']
        widgets = {
            'age': forms.NumberInput(attrs={'min': 18, 'max': 100, 'placeholder': 'Enter your age'}),
            'monthly_income': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 150000 MWK'}),
            'monthly_expenses': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 80000 MWK'}),
            'existing_debt': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 30000 MWK'}),
            'number_of_dependents': forms.NumberInput(attrs={'min': 0, 'placeholder': 'e.g., 2'}),
        }
        labels = {
            'monthly_income': 'Monthly Income (MWK)',
            'monthly_expenses': 'Monthly Expenses (MWK)',
            'existing_debt': 'Existing Debt (MWK)',
        }

class LoanApplicationForm(forms.ModelForm):
    class Meta:
        model = LoanApplication
        fields = ['amount', 'purpose', 'repayment_period']
        widgets = {
            'amount': forms.NumberInput(attrs={'min': 1000, 'max': 100000, 'step': '0.01', 'placeholder': 'e.g., 50000 MWK'}),
            'repayment_period': forms.NumberInput(attrs={'min': 1, 'max': 24, 'placeholder': 'In months'}),
            'purpose': forms.TextInput(attrs={'placeholder': 'e.g., Business'}),
        }
        labels = {
            'amount': 'Loan Amount (MWK)',
            'repayment_period': 'Repayment Period (Months)',
        }