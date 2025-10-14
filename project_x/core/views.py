from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile, LoanApplication, Transaction, MLModelPerformance
from .forms import UserProfileForm, LoanApplicationForm
import numpy as np
from django.utils import timezone
from django.db.models import Avg, Count, Sum
import subprocess

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
    return render(request, 'users/profile_update.html', {'form': form})

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile, LoanApplication, Transaction
from .forms import LoanApplicationForm
import numpy as np
from django.utils import timezone
import joblib


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
            
            application.save()
            
            # User feedback
            messages.success(request, decision['message'])
            return redirect('loan_status')
    else:
        form = LoanApplicationForm()
    
    return render(request, 'loans/loan_application.html', {'form': form})


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
        from django.db.models import Avg, Count
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
                'message': f'Your credit score is excellent ({credit_score}/1000), but your application requires manual review due to debt-to-income ratio. A loan officer will contact you within 48 hours.'
            }
    
    elif credit_score >= 700 and confidence >= 0.7:
        # Good credit + high confidence
        if dti_ratio < 0.35 and lti_ratio < 3 and loan_amount <= 50000:
            return {
                'status': 'Approved',
                'message': f'✅ Loan approved! Credit Score: {credit_score}/1000 (Good). Approved amount: MWK {loan_amount:.0f}. Disbursement in 24-48 hours.'
            }
        else:
            return {
                'status': 'Pending',
                'message': f'Your credit score is good ({credit_score}/1000). Application under review for final verification. Expected response: 2-3 business days.'
            }
    
    elif credit_score >= 650 and confidence >= 0.6:
        # Fair credit - needs review
        return {
            'status': 'Pending',
            'message': f'Your credit score is {credit_score}/1000 (Fair). Your application is under manual review by our credit team. You will receive a decision within 3-5 business days.'
        }
    
    elif credit_score >= 650 and confidence < 0.6:
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
    return render(request, 'loans/loan_status.html', {'applications': applications})


from django.shortcuts import render
from core.models import LoanApplication, MLModelPerformance
from django.utils import timezone
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from django.db.models import Sum

def admin_dashboard(request):
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
    return render(request, 'dashboard.html', context)

def retrain_model(request):
    if request.method == 'POST':
        try:
            # Load external data
            external_df = pd.read_csv('external_data.csv')
            X = external_df[['loan_amount', 'repayment_period', 'inflation_rate', 'unemployment_rate', 'default_rate']]
            y = external_df['credit_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            score_dist = {
                str(int(min(y_test))): int((y_pred >= int(min(y_test))).sum()),
                str(int(max(y_test))): int((y_pred >= int(max(y_test))).sum())
            }

            # Save performance
            from core.models import MLModelPerformance
            MLModelPerformance.objects.create(
                accuracy=mse,
                score_distribution=score_dist,
                last_trained=timezone.now(),
                notes=f"Training with only external CSV data (R²: {r2:.2f}, MSE: {mse:.2f})"
            )

            # Save model
            joblib.dump(model, 'credit_scoring_model.pkl')
            return render(request, 'dashboard.html', {'message': 'Model retrained successfully!'})
        except Exception as e:
            return render(request, 'dashboard.html', {'error': f'Error retraining model: {str(e)}'})
    return render(request, 'dashboard.html')


def _predict_credit_score(self, features):
    # Load Random Forest model
    import joblib
    try:
        model = joblib.load('credit_scoring_model.pkl')
        feature_array = np.array([[features['loan_amount'], features['repayment_period'],
                                  features.get('inflation_rate', 0), features.get('unemployment_rate', 0),
                                  features.get('default_rate', 0)]])
        # Predict probability of approval and scale to 0-1000
        score = int(model.predict(feature_array)[0])  # Direct score prediction
        return min(1000, max(0, score))
    except FileNotFoundError:
        # Fallback to simple rule-based scoring
        disposable_income = 100000 - 50000
        score = min(1000, max(0, int((disposable_income - 0) / features['loan_amount'] * 200)))
        return score

@login_required
def retrain_model(request):
    if not request.user.is_staff:
        return redirect('login')
    try:
        subprocess.run(['python', 'manage.py', 'train_ml_model'], check=True)
        messages.success(request, "Model retrained successfully!")
    except subprocess.CalledProcessError:
        messages.error(request, "Failed to retrain model. Check logs.")
    return redirect('admin_dashboard')