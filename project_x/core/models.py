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
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username}'s Profile"

class LoanApplication(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    purpose = models.CharField(max_length=100)
    repayment_period = models.PositiveIntegerField(help_text="In months")
    credit_score = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[('Pending', 'Pending'), ('Approved', 'Approved'), ('Rejected', 'Rejected')], default='Pending')
    submitted_at = models.DateTimeField(default=timezone.now)
    approved_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Loan {self.id} - {self.user_profile.user.username}"

class Transaction(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    transaction_type = models.CharField(max_length=20, choices=[('Deposit', 'Deposit'), ('Withdrawal', 'Withdrawal'), ('Transfer', 'Transfer')])
    transaction_date = models.DateTimeField(default=timezone.now)
    description = models.CharField(max_length=200, null=True, blank=True)

    def __str__(self):
        return f"{self.transaction_type} of {self.amount} MWK for {self.user_profile.user.username}"

class MLModelPerformance(models.Model):
    accuracy = models.FloatField()
    score_distribution = models.JSONField(default=dict)  # Store as JSON for histogram data
    last_trained = models.DateTimeField(default=timezone.now)
    notes = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"ML Performance - {self.last_trained}"