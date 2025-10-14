from django.core.management.base import BaseCommand
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from core.models import UserProfile, LoanApplication, MLModelPerformance, Transaction
import json
from django.utils import timezone
import joblib

class Command(BaseCommand):
    help = 'Trains the ML model for credit scoring using external data or merged data based on internal data volume'

    def handle(self, *args, **options):
        # Threshold for switching to merged data
        DATA_THRESHOLD = 50  # Adjust this value as needed

        # Step 1: Check internal data volume
        internal_count = LoanApplication.objects.count()
        use_merged_data = internal_count >= DATA_THRESHOLD

        # Step 2: Load External Dataset
        try:
            external_df = pd.read_csv('external_data.csv')
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("external_data.csv not found. Please add the file."))
            return

        required_columns = ['loan_amount', 'repayment_period', 'inflation_rate', 'unemployment_rate', 'default_rate', 'credit_score']
        if not all(col in external_df.columns for col in required_columns):
            self.stdout.write(self.style.ERROR("CSV must contain columns: loan_amount, repayment_period, inflation_rate, unemployment_rate, default_rate, credit_score"))
            return

        X_external = external_df[['loan_amount', 'repayment_period', 'inflation_rate', 'unemployment_rate', 'default_rate']]
        y_external = external_df['credit_score']

        if not use_merged_data:
            # Train with only external data
            X_train, X_test, y_train, y_test = train_test_split(X_external, y_external, test_size=0.2, random_state=42)
            notes = "Training with only external CSV data"
        else:
            # Step 3: Prepare Internal Dataset
            profiles = UserProfile.objects.all()
            applications = LoanApplication.objects.all()
            transactions = Transaction.objects.all()

            data = []
            for profile in profiles:
                app_data = applications.filter(user_profile=profile).first()
                trans_data = transactions.filter(user_profile=profile).aggregate(
                    avg_amount=models.Avg('amount'),
                    count=models.Count('id')
                )
                if app_data:
                    internal_features = {
                        'loan_amount': float(app_data.amount),
                        'repayment_period': app_data.repayment_period or 0,
                        'age': profile.age or 0,
                        'monthly_income': float(profile.monthly_income or 0),
                        'monthly_expenses': float(profile.monthly_expenses or 0),
                        'existing_debt': float(profile.existing_debt or 0),
                        'number_of_dependents': profile.number_of_dependents or 0,
                        'avg_transaction_amount': float(trans_data['avg_amount'] or 0),
                        'transaction_count': trans_data['count'] or 0,
                        'inflation_rate': 0,
                        'unemployment_rate': 0,
                        'default_rate': 0
                    }
                    # Match with external data by location
                    location = profile.location
                    external_features = {
                        'Lilongwe': {'inflation_rate': 8.5, 'unemployment_rate': 6.2, 'default_rate': 3.5},
                        'Blantyre': {'inflation_rate': 9.0, 'unemployment_rate': 7.0, 'default_rate': 4.0},
                        'Mzuzu': {'inflation_rate': 7.8, 'unemployment_rate': 5.8, 'default_rate': 3.0}
                    }.get(location, {'inflation_rate': 0, 'unemployment_rate': 0, 'default_rate': 0})
                    internal_features.update(external_features)
                    data.append((internal_features, app_data.credit_score))

            if not data:
                self.stdout.write(self.style.WARNING("No internal data available to merge. Falling back to external data."))
                X_train, X_test, y_train, y_test = train_test_split(X_external, y_external, test_size=0.2, random_state=42)
                notes = "Training with only external CSV data (no valid internal data)"
            else:
                # Merge internal and external data
                merged_df = pd.DataFrame([d[0] for d in data])
                y_merged = np.array([d[1] for d in data])
                X_train, X_test, y_train, y_test = train_test_split(merged_df, y_merged, test_size=0.2, random_state=42)
                notes = "Training with merged internal and external data"

        # Step 4: Train Random Forest Model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)

        # Step 5: Evaluate Model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Step 6: Store Performance
        # Calculate score distribution based on test set predictions
        score_dist = {
            str(int(min(y_test))): int((y_pred >= int(min(y_test))).sum()),
            str(int(max(y_test))): int((y_pred >= int(max(y_test))).sum())
        } if len(y_test) > 0 else {'590': 0, '840': 0}  # Fallback for empty test set
        MLModelPerformance.objects.create(
            accuracy=mse,
            score_distribution=score_dist,
            last_trained=timezone.now(),
            notes=notes + f" (R²: {r2:.2f}, MSE: {mse:.2f})"
        )

        # Save model for prediction
        joblib.dump(model, 'credit_scoring_model.pkl')

        self.stdout.write(self.style.SUCCESS(f"Model trained successfully with MSE: {mse:.2f}, R²: {r2:.2f}"))