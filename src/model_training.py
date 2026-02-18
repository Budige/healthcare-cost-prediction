"""
Healthcare Cost Prediction Model Training

Implements XGBoost and Random Forest ensemble with hyperparameter tuning,
cross-validation, and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareCostPredictor:
    """Ensemble model for predicting healthcare costs"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def create_ensemble_model(self) -> VotingRegressor:
        """
        Create ensemble of XGBoost and Random Forest
        
        Returns:
            VotingRegressor ensemble model
        """
        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        ensemble = VotingRegressor([
            ('xgboost', xgb),
            ('random_forest', rf)
        ])
        
        return ensemble
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from raw data
        
        Args:
            df: Raw patient data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 45, 65, 75, 100],
            labels=['18-44', '45-64', '65-74', '75+']
        )
        
        # BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Comorbidity risk score (Charlson Comorbidity Index simulation)
        comorbidity_weights = {
            'diabetes': 1,
            'heart_disease': 1,
            'copd': 1,
            'cancer': 2,
            'kidney_disease': 2
        }
        
        df['comorbidity_score'] = 0
        for condition, weight in comorbidity_weights.items():
            if condition in df.columns:
                df['comorbidity_score'] += df[condition] * weight
        
        # Hospital utilization intensity
        if 'hospital_stays' in df.columns and 'er_visits' in df.columns:
            df['utilization_intensity'] = df['hospital_stays'] * 2 + df['er_visits']
        
        # Prior cost tercile
        if 'prior_year_cost' in df.columns:
            df['prior_cost_tercile'] = pd.qcut(
                df['prior_year_cost'],
                q=3,
                labels=['Low', 'Medium', 'High']
            )
        
        logger.info(f"Engineered features. Shape: {df.shape}")
        return df
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Dict:
        """
        Train ensemble model with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training ensemble model...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Create and fit model
        self.model = self.create_ensemble_model()
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Training predictions
        y_train_pred = self.model.predict(X_train)
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std()
        }
        
        logger.info(f"Training complete. R² = {metrics['train_r2']:.3f}, "
                   f"MAE = ${metrics['train_mae']:.0f}")
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'test_r2': r2_score(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f"Test Results - R²: {metrics['test_r2']:.3f}, "
                   f"MAE: ${metrics['test_mae']:.0f}, "
                   f"MAPE: {metrics['test_mape']:.1f}%")
        
        return metrics
    
    def save_model(self, path: str = "models/"):
        """Save trained model and metadata"""
        Path(path).mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{path}/ensemble_model.pkl")
        
        # Save feature names
        with open(f"{path}/feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = "models/"):
        """Load trained model"""
        self.model = joblib.load(f"{path}/ensemble_model.pkl")
        
        with open(f"{path}/feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info(f"Model loaded from {path}")


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic healthcare data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 85, n_samples),
        'bmi': np.random.normal(28, 6, n_samples),
        'diabetes': np.random.binomial(1, 0.15, n_samples),
        'heart_disease': np.random.binomial(1, 0.12, n_samples),
        'copd': np.random.binomial(1, 0.08, n_samples),
        'cancer': np.random.binomial(1, 0.05, n_samples),
        'kidney_disease': np.random.binomial(1, 0.06, n_samples),
        'hospital_stays': np.random.poisson(1, n_samples),
        'er_visits': np.random.poisson(2, n_samples),
        'prior_year_cost': np.random.lognormal(8, 1.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with realistic relationships
    base_cost = 5000
    age_effect = (df['age'] - 18) * 100
    bmi_effect = (df['bmi'] - 25) * 50
    comorbidity_effect = (
        df['diabetes'] * 3000 + 
        df['heart_disease'] * 4000 +
        df['copd'] * 3500 +
        df['cancer'] * 8000 +
        df['kidney_disease'] * 5000
    )
    utilization_effect = df['hospital_stays'] * 5000 + df['er_visits'] * 1000
    prior_effect = df['prior_year_cost'] * 0.3
    noise = np.random.normal(0, 2000, n_samples)
    
    df['healthcare_cost'] = (
        base_cost + age_effect + bmi_effect + 
        comorbidity_effect + utilization_effect + 
        prior_effect + noise
    ).clip(lower=0)
    
    return df


def main():
    """Main training pipeline"""
    
    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(n_samples=10000)
    
    # Initialize predictor
    predictor = HealthcareCostPredictor()
    
    # Engineer features
    df = predictor.engineer_features(df)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != 'healthcare_cost']
    # Convert categorical to dummy variables
    df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)
    
    X = df_encoded
    y = df['healthcare_cost']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    train_metrics = predictor.train(X_train, y_train)
    
    # Evaluate
    test_metrics = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    # Print summary
    print("\n=== Model Training Complete ===")
    print(f"Train R²: {train_metrics['train_r2']:.3f}")
    print(f"Test R²: {test_metrics['test_r2']:.3f}")
    print(f"Test MAE: ${test_metrics['test_mae']:.0f}")
    print(f"Test MAPE: {test_metrics['test_mape']:.1f}%")
    print(f"\nModel saved to models/ensemble_model.pkl")


if __name__ == "__main__":
    main()
