"""
Data Preprocessing & Feature Engineering Pipeline
-------------------------------------------------
Handles all data cleaning, transformation, and feature engineering
for the Customer Churn Prediction system.

Key responsibilities:
- Missing value handling
- Categorical encoding (Label + One-Hot)
- Feature scaling
- Class imbalance handling via SMOTE
- Feature engineering for business insights
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ChurnDataProcessor:
    """
    End-to-end data processor for churn prediction.
    
    Handles the complete preprocessing pipeline including:
    - Data cleaning and type conversion
    - Missing value imputation
    - Feature engineering
    - Encoding and scaling
    - Train/test splitting
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.binary_cols = None
        self._fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data:
        - Handle TotalCharges blank values
        - Convert data types
        - Remove customerID (not predictive)
        """
        df = df.copy()
        
        # TotalCharges has some blank values (new customers)
        # Convert to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with MonthlyCharges (new customers)
        mask = df['TotalCharges'].isna()
        df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
        
        print(f"Cleaned {mask.sum()} missing TotalCharges values")
        
        # Drop customerID - not useful for prediction
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-relevant features that improve model performance.
        
        New features:
        - tenure_group: Categorical tenure buckets
        - avg_monthly_spend: TotalCharges / tenure
        - service_count: Number of additional services
        - has_premium_support: Security + Tech Support
        - has_streaming: TV + Movies bundle
        - charge_per_service: Value metric
        """
        df = df.copy()
        
        # Tenure groups (aligned with typical customer lifecycle)
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+']
        ).astype(str)
        
        # Average monthly spend (handles tenure=0)
        df['avg_monthly_spend'] = df.apply(
            lambda x: x['TotalCharges'] / max(x['tenure'], 1), axis=1
        )
        
        # Count of additional services subscribed
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['service_count'] = df[service_cols].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        # Premium support bundle indicator
        df['has_premium_support'] = (
            (df['OnlineSecurity'] == 'Yes') & (df['TechSupport'] == 'Yes')
        ).astype(int)
        
        # Streaming bundle indicator
        df['has_streaming_bundle'] = (
            (df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')
        ).astype(int)
        
        # High value customer flag (long tenure + high spend)
        df['is_high_value'] = (
            (df['tenure'] >= 24) & (df['MonthlyCharges'] >= 70)
        ).astype(int)
        
        # Contract risk score (month-to-month with short tenure is risky)
        df['contract_risk'] = (
            (df['Contract'] == 'Month-to-month') & (df['tenure'] <= 12)
        ).astype(int)
        
        # Auto-pay indicator (lower churn typically)
        df['has_auto_pay'] = df['PaymentMethod'].apply(
            lambda x: 1 if 'automatic' in str(x).lower() else 0
        )
        
        print(f"Engineered {9} new features")
        return df
    
    def identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify categorical, numerical, and binary columns."""
        # Binary columns (Yes/No only)
        self.binary_cols = ['Partner', 'Dependents', 'PhoneService', 
                           'PaperlessBilling', 'Churn']
        
        # Categorical columns with multiple values
        self.categorical_cols = ['gender', 'MultipleLines', 'InternetService',
                                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies',
                                'Contract', 'PaymentMethod', 'tenure_group']
        
        # Numerical columns
        self.numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 
                              'TotalCharges', 'avg_monthly_spend', 'service_count',
                              'has_premium_support', 'has_streaming_bundle',
                              'is_high_value', 'contract_risk', 'has_auto_pay']
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Strategy:
        - Binary columns: Label encoding (0/1)
        - Multi-class categorical: One-hot encoding
        - Numerical: Keep as-is (will be scaled separately)
        """
        df = df.copy()
        
        # Label encode binary columns
        for col in self.binary_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # One-hot encode multi-class categorical columns
        existing_cats = [c for c in self.categorical_cols if c in df.columns]
        df = pd.get_dummies(df, columns=existing_cats, drop_first=False)
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numerical features using StandardScaler."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Full preprocessing pipeline - fit and transform training data.
        
        Returns:
        --------
        X : np.ndarray - Preprocessed features
        y : np.ndarray - Target variable
        feature_names : List[str] - Names of all features
        """
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Identify column types
        self.identify_column_types(df)
        
        # Encode features
        df = self.encode_features(df, fit=True)
        
        # Separate target
        y = df['Churn'].values
        X = df.drop('Churn', axis=1)
        
        # Store feature names before scaling
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        self._fitted = True
        
        return X_scaled, y, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        Use this for inference on new customers.
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_features(df, fit=False)
        
        # Ensure same columns as training
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_names]
        X_scaled = self.scale_features(X, fit=False)
        
        return X_scaled
    
    def save(self, filepath: str) -> None:
        """Save preprocessor state to disk."""
        state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'binary_cols': self.binary_cols,
            '_fitted': self._fitted
        }
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str) -> 'ChurnDataProcessor':
        """Load preprocessor state from disk."""
        state = joblib.load(filepath)
        self.label_encoders = state['label_encoders']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.categorical_cols = state['categorical_cols']
        self.numerical_cols = state['numerical_cols']
        self.binary_cols = state['binary_cols']
        self._fitted = state['_fitted']
        print(f"Preprocessor loaded from {filepath}")
        return self
    
    @classmethod
    def from_file(cls, filepath: str) -> 'ChurnDataProcessor':
        """Create a preprocessor instance from a saved file."""
        instance = cls()
        return instance.load(filepath)


def prepare_train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets with stratification.
    
    Stratification ensures equal churn proportion in both sets,
    which is critical for imbalanced datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance.
    
    SMOTE (Synthetic Minority Over-sampling Technique) creates
    synthetic samples of the minority class (churned customers)
    to balance the dataset.
    
    IMPORTANT: Only apply to training data, never to test data!
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"\nSMOTE applied:")
        print(f"Before: {len(X_train)} samples (Churn: {y_train.sum()})")
        print(f"After: {len(X_resampled)} samples (Churn: {y_resampled.sum()})")
        
        return X_resampled, y_resampled
        
    except ImportError:
        print("Warning: imbalanced-learn not installed. Using class weights instead.")
        return X_train, y_train


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data.
    
    Alternative to SMOTE - can be passed to model training
    to penalize misclassification of minority class more heavily.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


if __name__ == '__main__':
    # Test the preprocessing pipeline
    from data_generator import generate_telco_dataset
    
    # Generate test data
    df = generate_telco_dataset(n_samples=1000)
    
    # Initialize processor
    processor = ChurnDataProcessor()
    
    # Fit and transform
    X, y, features = processor.fit_transform(df)
    
    print(f"\nProcessed data shape: {X.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Test train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # Test class weights
    weights = get_class_weights(y_train)
    print(f"\nClass weights: {weights}")
