# core/models.py (updated)
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')], null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)
    employment_status = models.CharField(max_length=50, choices=[('Employed', 'Employed'), ('Self-employed', 'Self-employed'), ('Unemployed', 'Unemployed')], null=True, blank=True)
    monthly_income = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    monthly_expenses = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    existing_debt = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    number_of_dependents = models.PositiveIntegerField(null=True, blank=True)
    mobile_usage_monthly = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True, help_text="Average monthly mobile usage (MWK)")
    utility_bills_monthly = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True, help_text="Average monthly utility bills (MWK)")
    proof_documents = models.FileField(upload_to='user_proofs/', null=True, blank=True, help_text="Upload proof (e.g., bank statement, utility bill)")
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username}'s Profile"

# core/models.py (updated excerpt)
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# ... (keep UserProfile, LoanApplication, etc.)

class Transaction(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    transaction_type = models.CharField(max_length=20, choices=[('Deposit', 'Deposit'), ('Withdrawal', 'Withdrawal'), ('Transfer', 'Transfer'), ('Mobile Top-up', 'Mobile Top-up'), ('Utility Payment', 'Utility Payment')])
    transaction_date = models.DateTimeField(default=timezone.now)
    description = models.CharField(max_length=200, null=True, blank=True)
    reference_number = models.CharField(max_length=50, null=True, blank=True, help_text="Unique transaction reference number (e.g., from bank)")  # New field
    source = models.CharField(max_length=100, null=True, blank=True, help_text="Source of transaction (e.g., Airtel, TNM, Bank)")  # New field

    def __str__(self):
        return f"{self.transaction_type} of {self.amount} MWK for {self.user_profile.user.username} (Ref: {self.reference_number or 'N/A'})"

# ... (keep LoanApplication, Repayment, MLModelPerformance as is)

class LoanApplication(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    purpose = models.CharField(max_length=100)
    repayment_period = models.PositiveIntegerField(help_text="In months")
    credit_score = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[('Pending', 'Pending'), ('Approved', 'Approved'), ('Rejected', 'Rejected'), ('Paid', 'Paid')], default='Pending')
    outstanding_amount = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    submitted_at = models.DateTimeField(default=timezone.now)
    approved_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Loan {self.id} - {self.user_profile.user.username}"



class Repayment(models.Model):
    loan = models.ForeignKey(LoanApplication, on_delete=models.CASCADE, related_name='repayments')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Repayment of {self.amount} MWK for Loan {self.loan.id}"

class MLModelPerformance(models.Model):
    accuracy = models.FloatField()
    score_distribution = models.JSONField(default=dict)  # Store as JSON for histogram data
    last_trained = models.DateTimeField(default=timezone.now)
    notes = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"ML Performance - {self.last_trained}"