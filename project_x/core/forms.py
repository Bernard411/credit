# core/forms.py (updated with RepaymentForm)
# core/forms.py (updated)
from django import forms
from .models import UserProfile, LoanApplication, Transaction

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'gender', 'location', 'employment_status', 'monthly_income', 'monthly_expenses', 'existing_debt', 'number_of_dependents', 'mobile_usage_monthly', 'utility_bills_monthly', 'proof_documents']
        widgets = {
            'age': forms.NumberInput(attrs={'min': 18, 'max': 100, 'placeholder': 'Enter your age'}),
            'monthly_income': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 150000 MWK'}),
            'monthly_expenses': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 80000 MWK'}),
            'existing_debt': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 30000 MWK'}),
            'number_of_dependents': forms.NumberInput(attrs={'min': 0, 'placeholder': 'e.g., 2'}),
            'mobile_usage_monthly': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 5000 MWK'}),
            'utility_bills_monthly': forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 10000 MWK'}),
            'proof_documents': forms.ClearableFileInput(attrs={'multiple': False}),
        }
        labels = {
            'monthly_income': 'Monthly Income (MWK)',
            'monthly_expenses': 'Monthly Expenses (MWK)',
            'existing_debt': 'Existing Debt (MWK)',
            'mobile_usage_monthly': 'Average Monthly Mobile Usage (MWK)',
            'utility_bills_monthly': 'Average Monthly Utility Bills (MWK)',
            'proof_documents': 'Supporting Documents',
        }

# core/forms.py (updated excerpt)
from django import forms
from .models import UserProfile, LoanApplication, Transaction

# ... (keep UserProfileForm, LoanApplicationForm, etc.)

class TransactionForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = ['amount', 'transaction_type', 'description', 'transaction_date', 'reference_number', 'source']
        widgets = {
            'transaction_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'description': forms.TextInput(attrs={'placeholder': 'e.g., Salary deposit or Utility payment'}),
            'reference_number': forms.TextInput(attrs={'placeholder': 'e.g., TXN123456789'}),
            'source': forms.TextInput(attrs={'placeholder': 'e.g., Airtel, TNM, Bank'}),
        }

# ... (keep LoanApplicationForm, RepaymentForm, LoginForm as is)

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

class RepaymentForm(forms.Form):
    amount = forms.DecimalField(
        min_value=0.01,
        max_digits=12,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'e.g., 10000 MWK'})
    )
    
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.core.exceptions import ValidationError

class LoginForm(forms.Form):
    username_or_email = forms.CharField(
        label='Username or Email',
        widget=forms.TextInput(attrs={
            'class': 'form-input w-full focus:outline-none focus:ring-2 focus:ring-blue-500',
            'placeholder': 'Enter your username or email'
        })
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-input w-full focus:outline-none focus:ring-2 focus:ring-blue-500',
            'placeholder': 'Enter your password'
        })
    )

    def clean_username_or_email(self):
        username_or_email = self.cleaned_data.get('username_or_email')
        if '@' in username_or_email:
            try:
                user = User.objects.get(email=username_or_email)
                return user.username
            except User.DoesNotExist:
                raise ValidationError("No account found with this email address.")
        return username_or_email

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username_or_email')
        password = cleaned_data.get('password')

        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise ValidationError("Invalid username/email or password.")
        return cleaned_data