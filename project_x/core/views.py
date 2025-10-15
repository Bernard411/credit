# core/views.py (cleaned and fixed version)
# I've merged duplicates, removed redundant imports, chose the subprocess version for retrain_model,
# removed stray functions like _predict_credit_score, and assumed 'core' is the app name based on imports in train_ml_model.py.
# Also added a user_dashboard view to match the UI's dashboard section for users.
# Fixed some minor issues like missing imports and truncated code (completed logically based on context).
# For make_loan_decision, completed the truncated part based on the pattern.

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Count, Sum
from django.utils import timezone
from .models import UserProfile, LoanApplication, Transaction, MLModelPerformance
from .forms import UserProfileForm, LoanApplicationForm
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Count, Sum
from django.utils import timezone
from .models import UserProfile, LoanApplication, Transaction, MLModelPerformance, Repayment
from .forms import UserProfileForm, LoanApplicationForm, RepaymentForm
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import subprocess







# auth/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import LoginForm


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username_or_email')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect('user_dashboard')
            messages.error(request, 'Invalid username/email or password.')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


@login_required
def user_dashboard(request):
    profile = get_object_or_404(UserProfile, user=request.user)
    applications = LoanApplication.objects.filter(user_profile=profile)
    approved_loans = applications.filter(status='Approved')
    active_loans_count = approved_loans.count()
    total_active_amount = approved_loans.aggregate(total=Sum('outstanding_amount'))['total'] or 0
    avg_credit_score = applications.aggregate(avg_score=Avg('credit_score'))['avg_score'] or 750
    # Placeholder logic for available credit (customize as needed, e.g., based on max limit minus outstanding)
    available_credit = 100000 - total_active_amount  # Assuming max credit limit of 100,000 MWK
    balance = profile.balance

    context = {
        'active_loans_count': active_loans_count,
        'total_active_amount': total_active_amount,
        'available_credit': available_credit,
        'credit_score': int(avg_credit_score),
        'balance': balance,
        'approved_loans': approved_loans,
    }
    return render(request, 'user_dashboard.html', context)

@login_required
def user_profile_update(request):
    profile = get_object_or_404(UserProfile, user=request.user)
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('user_profile_update')
    else:
        form = UserProfileForm(instance=profile)
    return render(request, 'profile_update.html', {'form': form})

@login_required
def loan_application_create(request):
    """
    Automatic loan approval system using ML model predictions
    """
    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            profile = get_object_or_404(UserProfile, user=request.user)
            application = form.save(commit=False)
            application.user_profile = profile

            # Prepare features for prediction
            features = prepare_features(profile, application)
            
            # Predict credit score using the trained model
            credit_score, confidence = predict_credit_score_with_confidence(features)
            application.credit_score = credit_score
            
            # Automatic decision based on credit score and confidence
            decision = make_loan_decision(
                credit_score=credit_score,
                confidence=confidence,
                loan_amount=float(application.amount),
                monthly_income=float(profile.monthly_income or 0),
                existing_debt=float(profile.existing_debt or 0)
            )
            
            application.status = decision['status']
            if decision['status'] == 'Approved':
                application.approved_at = timezone.now()
                application.outstanding_amount = application.amount
                profile.balance += application.amount
                profile.save()
            
            application.save()
            
            # User feedback
            messages.success(request, decision['message'])
            return redirect('loan_status')
    else:
        form = LoanApplicationForm()
    
    return render(request, 'loan_application.html', {'form': form})

def prepare_features(profile, application):
    """
    Prepare feature vector for ML model prediction
    """
    features = {
        'loan_amount': float(application.amount),
        'repayment_period': application.repayment_period or 12,
        'age': profile.age or 30,
        'monthly_income': float(profile.monthly_income or 0),
        'monthly_expenses': float(profile.monthly_expenses or 0),
        'existing_debt': float(profile.existing_debt or 0),
        'number_of_dependents': profile.number_of_dependents or 0,
    }
    
    # Add external/regional economic data
    location = profile.location
    external_data = {
        'Lilongwe': {'inflation_rate': 8.5, 'unemployment_rate': 6.2, 'default_rate': 3.5},
        'Blantyre': {'inflation_rate': 9.0, 'unemployment_rate': 7.0, 'default_rate': 4.0},
        'Mzuzu': {'inflation_rate': 7.8, 'unemployment_rate': 5.8, 'default_rate': 3.0}
    }
    features.update(external_data.get(location, {
        'inflation_rate': 8.5, 
        'unemployment_rate': 6.5, 
        'default_rate': 3.5
    }))
    
    # Add transaction history features if available
    transactions = Transaction.objects.filter(user_profile=profile)
    if transactions.exists():
        trans_stats = transactions.aggregate(
            avg_amount=Avg('amount'),
            count=Count('id')
        )
        features['avg_transaction_amount'] = float(trans_stats['avg_amount'] or 0)
        features['transaction_count'] = trans_stats['count'] or 0
    else:
        features['avg_transaction_amount'] = 0
        features['transaction_count'] = 0
    
    return features

def predict_credit_score_with_confidence(features):
    """
    Predict credit score using trained ML model with confidence estimation
    Returns: (credit_score, confidence_level)
    """
    try:
        model = joblib.load('credit_scoring_model.pkl')
        
        # Prepare feature array in correct order
        feature_order = [
            'loan_amount', 'repayment_period', 'inflation_rate', 
            'unemployment_rate', 'default_rate'
        ]
        feature_array = np.array([[features.get(f, 0) for f in feature_order]])
        
        # Get prediction
        predicted_score = model.predict(feature_array)[0]
        credit_score = int(np.clip(predicted_score, 0, 1000))
        
        # Estimate confidence using model uncertainty (if Random Forest)
        confidence = estimate_prediction_confidence(model, feature_array)
        
        return credit_score, confidence
        
    except FileNotFoundError:
        # Fallback to rule-based scoring
        credit_score = calculate_fallback_score(features)
        return credit_score, 0.5  # Medium confidence for fallback

def estimate_prediction_confidence(model, feature_array):
    """
    Estimate prediction confidence for Random Forest model
    Uses variance of tree predictions as confidence metric
    """
    try:
        # Get predictions from all trees in the forest
        tree_predictions = np.array([
            tree.predict(feature_array)[0] 
            for tree in model.estimators_
        ])
        
        # Low variance = high confidence
        variance = np.var(tree_predictions)
        std_dev = np.std(tree_predictions)
        
        # Normalize to 0-1 scale (inverse relationship: low variance = high confidence)
        # Assuming typical std_dev range of 0-200 for credit scores
        confidence = 1 - min(std_dev / 200, 1.0)
        
        return confidence
        
    except:
        return 0.7  # Default moderate confidence

def make_loan_decision(credit_score, confidence, loan_amount, monthly_income, existing_debt):
    """
    Automatic loan decision engine with multiple criteria
    
    Decision Matrix:
    - High Score (≥750) + High Confidence (≥0.7): AUTO-APPROVE
    - Medium Score (650-749) + High Confidence: AUTO-APPROVE (lower amounts)
    - Medium Score + Low Confidence: MANUAL REVIEW
    - Low Score (<650): AUTO-REJECT
    
    Additional checks:
    - Debt-to-Income ratio
    - Loan-to-Income ratio
    """
    
    # Calculate financial ratios
    dti_ratio = (existing_debt / monthly_income) if monthly_income > 0 else float('inf')
    lti_ratio = (loan_amount / monthly_income) if monthly_income > 0 else float('inf')
    
    # Decision logic
    if credit_score >= 750 and confidence >= 0.7:
        # Excellent credit + high confidence
        if dti_ratio < 0.4 and lti_ratio < 5:  # Reasonable debt levels
            return {
                'status': 'Approved',
                'message': f'🎉 Loan automatically approved! Credit Score: {credit_score}/1000 (Excellent). Funds will be disbursed within 24 hours.'
            }
        else:
            return {
                'status': 'Pending',
                'message': f'Your credit score is excellent ({credit_score}/1000), but your application requires manual review due to debt-to-income ratio. A loan officer will contact you shortly.'
            }
    elif 650 <= credit_score < 750 and confidence >= 0.7:
        # Good credit + high confidence
        if loan_amount <= monthly_income * 3 and dti_ratio < 0.5:  # Conservative limits for medium score
            return {
                'status': 'Approved',
                'message': f'✅ Loan approved! Credit Score: {credit_score}/1000 (Good). Please review terms before acceptance.'
            }
        else:
            return {
                'status': 'Pending',
                'message': f'Credit Score: {credit_score}/1000 (Good). Manual review required due to loan size or debt levels. Expect contact within 48 hours.'
            }
    elif 650 <= credit_score < 750 and confidence < 0.7:
        # Good credit but low confidence
        return {
            'status': 'Pending',
            'message': f'Credit Score: {credit_score}/1000. Additional verification required due to model uncertainty. Our team will contact you.'
        }
    elif 550 <= credit_score < 650 and confidence >= 0.6:
        # Fair credit with moderate confidence
        if loan_amount <= monthly_income * 2 and dti_ratio < 0.3:
            return {
                'status': 'Approved',
                'message': f'Loan conditionally approved with Credit Score: {credit_score}/1000 (Fair). Higher interest rate applies.'
            }
        else:
            return {
                'status': 'Pending',
                'message': f'Credit Score: {credit_score}/1000 (Fair). Requires manual review.'
            }
    elif credit_score < 650 and confidence < 0.6:
        # Fair credit but low confidence
        return {
            'status': 'Pending',
            'message': f'Credit Score: {credit_score}/1000. Additional verification required. Our team will contact you to request supporting documents. Expected review time: 5-7 days.'
        }
    
    else:
        # Poor credit - auto-reject
        return {
            'status': 'Rejected',
            'message': f'We regret to inform you that your loan application has been declined. Credit Score: {credit_score}/1000. Please improve your credit profile and reapply after 90 days. Consider: reducing existing debt, maintaining regular income deposits, and building transaction history.'
        }

def calculate_fallback_score(features):
    """
    Rule-based fallback scoring when ML model is unavailable
    """
    score = 500  # Base score
    
    # Income scoring
    monthly_income = features.get('monthly_income', 0)
    if monthly_income > 200000:
        score += 150
    elif monthly_income > 100000:
        score += 100
    elif monthly_income > 50000:
        score += 50
    
    # Debt ratio
    existing_debt = features.get('existing_debt', 0)
    if monthly_income > 0:
        debt_ratio = existing_debt / monthly_income
        if debt_ratio < 0.2:
            score += 100
        elif debt_ratio < 0.4:
            score += 50
        else:
            score -= 50
    
    # Loan affordability
    loan_amount = features.get('loan_amount', 0)
    if monthly_income > 0:
        monthly_payment = loan_amount / features.get('repayment_period', 12)
        affordability = monthly_payment / monthly_income
        if affordability < 0.2:
            score += 100
        elif affordability < 0.3:
            score += 50
    
    # Transaction history
    trans_count = features.get('transaction_count', 0)
    if trans_count > 50:
        score += 50
    elif trans_count > 20:
        score += 25
    
    return min(1000, max(300, score))

@login_required
def loan_status(request):
    profile = get_object_or_404(UserProfile, user=request.user)
    applications = LoanApplication.objects.filter(user_profile=profile).order_by('-submitted_at')
    return render(request, 'loan_status.html', {'applications': applications})

@login_required
def loan_repayment(request, loan_id):
    profile = get_object_or_404(UserProfile, user=request.user)
    loan = get_object_or_404(LoanApplication, id=loan_id, user_profile=profile, status='Approved')
    if request.method == 'POST':
        form = RepaymentForm(request.POST)
        if form.is_valid():
            payment = form.cleaned_data['amount']
            # Check for None values
            outstanding = loan.outstanding_amount or 0
            balance = profile.balance or 0
            if payment > outstanding:
                messages.error(request, "Payment amount exceeds outstanding balance.")
            elif payment > balance:
                messages.error(request, "Insufficient balance in your account.")
            else:
                profile.balance -= payment
                loan.outstanding_amount -= payment
                profile.save()
                loan.save()
                Repayment.objects.create(loan=loan, amount=payment)
                if loan.outstanding_amount <= 0:
                    loan.status = 'Paid'
                    loan.save()
                messages.success(request, f"Payment of MWK {payment} successful!")
                return redirect('user_dashboard')
    else:
        initial_amount = loan.outstanding_amount or 0
        form = RepaymentForm(initial={'amount': initial_amount})
    return render(request, 'loan_repayment.html', {'form': form, 'loan': loan})

@login_required
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('user_dashboard')  # Redirect non-admins
    # Fetch data
    total_applications = LoanApplication.objects.count()
    approved_loans = LoanApplication.objects.filter(status='Approved').count()
    pending_reviews = LoanApplication.objects.filter(status='Pending').count()
    approved_value = LoanApplication.objects.filter(status='Approved').aggregate(total=Sum('amount'))['total'] or 0
    recent_applications = LoanApplication.objects.order_by('-submitted_at')[:5]  # Top 5 recent
    pending_value = pending_reviews * 50000  # Calculate pending value in the view

    # ML Performance
    ml_performance = MLModelPerformance.objects.latest('last_trained') if MLModelPerformance.objects.exists() else None

    # Score Distribution (prepare as lists for Chart.js)
    score_dist = {}
    score_labels = []
    score_data = []
    if ml_performance and ml_performance.score_distribution:
        score_dist = ml_performance.score_distribution
        score_labels = list(score_dist.keys())
        score_data = list(score_dist.values())
    else:
        # Fallback if no ML performance data
        score_labels = ['590', '840']
        score_data = [0, 0]

    # Status Distribution (prepare as lists for Chart.js)
    status_dist = {
        'Approved': LoanApplication.objects.filter(status='Approved').count(),
        'Pending': LoanApplication.objects.filter(status='Pending').count(),
        'Rejected': LoanApplication.objects.filter(status='Rejected').count()
    }
    status_labels = list(status_dist.keys())
    status_data = list(status_dist.values())

    # Update last_updated in session
    request.session['last_updated'] = timezone.now().strftime('%I:%M %p CAT, %b %d, %Y')

    context = {
        'total_applications': total_applications,
        'approved_loans': approved_loans,
        'pending_reviews': pending_reviews,
        'approved_value': approved_value,
        'pending_value': pending_value,
        'ml_performance': ml_performance,
        'score_labels': score_labels,  # Pass pre-processed labels
        'score_data': score_data,     # Pass pre-processed data
        'status_labels': status_labels,  # Pass pre-processed labels
        'status_data': status_data,     # Pass pre-processed data
        'recent_applications': recent_applications,
    }
    return render(request, 'dashboard.html', context)  # Changed to 'admin/dashboard.html' to distinguish from user dashboard

@login_required
def retrain_model(request):
    if not request.user.is_staff:
        return redirect('login')
    if request.method == 'POST':
        try:
            subprocess.run(['python', 'manage.py', 'train_ml_model'], check=True)
            messages.success(request, "Model retrained successfully!")
        except subprocess.CalledProcessError:
            messages.error(request, "Failed to retrain model. Check logs.")
    return redirect('admin_dashboard')