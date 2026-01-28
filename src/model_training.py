"""
Model Training Module
---------------------
Trains and evaluates churn prediction models.

Models implemented:
1. Logistic Regression (baseline - interpretable)
2. Random Forest (ensemble - better performance)
3. XGBoost (gradient boosting - state of the art) [optional]

Focus: Recall & Precision over Accuracy
- False Negatives are costly (missing churners = lost customers)
- False Positives have moderate cost (unnecessary retention spend)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score,
    recall_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
from typing import Dict, Tuple, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """
    Handles training, evaluation, and comparison of churn prediction models.
    
    Business Context:
    -----------------
    The cost of a False Negative (missing a churner) is typically 5-25x
    higher than a False Positive (flagging a loyal customer).
    
    Rationale:
    - Lost customer = Lost lifetime value ($500-$2000+ per customer)
    - Unnecessary retention offer = $20-50 per customer
    
    Therefore, we optimize for RECALL while maintaining reasonable precision.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        class_weight: str = 'balanced'
    ) -> LogisticRegression:
        """
        Train Logistic Regression baseline model.
        
        Why use this:
        - Highly interpretable (coefficients show feature importance)
        - Fast training and inference
        - Good baseline to compare against
        - Provides probability calibration
        """
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            random_state=self.random_state,
            class_weight=class_weight,  # Handle imbalance
            max_iter=1000,
            solver='lbfgs',
            C=1.0  # Regularization strength
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        print("Training complete.")
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_weight: str = 'balanced',
        n_estimators: int = 200,
        max_depth: int = 15,
        tune_hyperparams: bool = False
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Why use this:
        - Handles non-linear relationships
        - Built-in feature importance
        - Robust to outliers
        - Less prone to overfitting than single trees
        """
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight=class_weight,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=3, scoring='f1',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight=class_weight,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Training complete.")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale_pos_weight: float = None
    ):
        """
        Train XGBoost model (if available).
        
        Why use this:
        - State-of-the-art performance on tabular data
        - Handles missing values natively
        - Built-in regularization
        - Feature importance available
        """
        try:
            from xgboost import XGBClassifier
            
            print("\n" + "="*50)
            print("Training XGBoost...")
            print("="*50)
            
            # Calculate scale_pos_weight for imbalance
            if scale_pos_weight is None:
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                scale_pos_weight = neg_count / pos_count
            
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            self.models['xgboost'] = model
            print("Training complete.")
            return model
            
        except ImportError:
            print("XGBoost not installed. Skipping...")
            return None
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Metrics computed:
        - Confusion Matrix
        - Precision, Recall, F1-Score
        - ROC-AUC
        - Precision-Recall AUC (better for imbalanced data)
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'model_name': model_name,
            'threshold': threshold,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        # Print results
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted No  Predicted Yes")
        print(f"Actual No          {tn:>8}        {fp:>8}")
        print(f"Actual Yes         {fn:>8}        {tp:>8}")
        
        print(f"\nKey Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}  <- Primary metric!")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.3f}")
        
        # Business interpretation
        print(f"\nBusiness Interpretation:")
        print(f"  Churners correctly identified: {tp} ({metrics['recall']:.1%})")
        print(f"  Churners missed (False Negatives): {fn}")
        print(f"  False alarms (False Positives): {fp}")
        
        self.results[model_name] = metrics
        return metrics
    
    def find_optimal_threshold(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fn_cost: float = 500,  # Cost of missing a churner
        fp_cost: float = 50    # Cost of false alarm
    ) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold based on business costs.
        
        The default 0.5 threshold often isn't optimal. We find the threshold
        that minimizes total business cost.
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_cost = float('inf')
        results = []
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Total cost calculation
            total_cost = (fn * fn_cost) + (fp * fp_cost)
            
            results.append({
                'threshold': threshold,
                'fn': fn,
                'fp': fp,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'total_cost': total_cost
            })
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        print(f"\nOptimal threshold analysis (FN cost=${fn_cost}, FP cost=${fp_cost}):")
        print(f"  Optimal threshold: {best_threshold:.2f}")
        print(f"  Minimum total cost: ${best_cost:,.0f}")
        
        return best_threshold, results
    
    def get_feature_importance(
        self,
        model,
        feature_names: List[str],
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Works with:
        - Logistic Regression: Uses absolute coefficients
        - Random Forest/XGBoost: Uses built-in importance
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model doesn't have feature importance attribute")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Normalize to percentages
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df.head(top_n)
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models and select the best one."""
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        comparison = pd.DataFrame(self.results).T
        comparison = comparison[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison.round(3).to_string())
        
        # Select best model based on F1 score (balance of precision and recall)
        best_idx = comparison['f1_score'].idxmax()
        self.best_model_name = best_idx
        self.best_model = self.models[best_idx]
        
        print(f"\nBest model: {best_idx} (F1: {comparison.loc[best_idx, 'f1_score']:.3f})")
        
        return comparison
    
    def save_model(self, model, filepath: str) -> None:
        """Save trained model to disk."""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON."""
        # Convert numpy types to native Python types
        results_clean = {}
        for model_name, metrics in self.results.items():
            results_clean[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"Results saved to {filepath}")


def calculate_business_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    avg_customer_value: float = 1200,
    retention_cost: float = 50,
    retention_success_rate: float = 0.3
) -> Dict[str, float]:
    """
    Calculate the business impact of the churn prediction model.
    
    Assumptions:
    - Average customer lifetime value: $1,200
    - Cost of retention campaign: $50 per customer
    - Retention success rate: 30% (industry average)
    
    Returns dictionary with financial metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # True Positives: Churners we identify and can try to retain
    # Assume 30% success rate in retention
    saved_customers = tp * retention_success_rate
    revenue_saved = saved_customers * avg_customer_value
    
    # Cost of retention campaigns (for all predicted churners)
    retention_campaign_cost = (tp + fp) * retention_cost
    
    # Lost revenue from missed churners (False Negatives)
    lost_revenue = fn * avg_customer_value
    
    # Net impact
    net_impact = revenue_saved - retention_campaign_cost
    
    # Without model: All churners lost
    baseline_loss = (tp + fn) * avg_customer_value
    
    # Model improvement
    model_value = baseline_loss - (lost_revenue + retention_campaign_cost - revenue_saved)
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'saved_customers': saved_customers,
        'revenue_saved': revenue_saved,
        'retention_cost': retention_campaign_cost,
        'lost_revenue_from_fn': lost_revenue,
        'net_impact': net_impact,
        'baseline_loss': baseline_loss,
        'model_value': model_value,
        'roi_percentage': (model_value / retention_campaign_cost * 100) if retention_campaign_cost > 0 else 0
    }


def explain_false_negatives():
    """
    Business explanation of False Negative cost.
    
    Returns explanation string for documentation/UI.
    """
    explanation = """
    FALSE NEGATIVES IN CHURN PREDICTION
    ====================================
    
    A False Negative occurs when our model predicts a customer will STAY,
    but they actually CHURN.
    
    BUSINESS IMPACT:
    ----------------
    1. Lost Revenue: The customer's future payments are lost
       - Average customer value: $1,200/year
       - Multi-year impact: Could be $3,000-$5,000+ over time
    
    2. Acquisition Cost Wasted: We spent money to acquire this customer
       - Average acquisition cost: $300-$500
       - That investment is now lost
    
    3. Competitive Damage: Customer may go to competitor
       - Word of mouth: Unhappy customers tell 9-15 people
       - Online reviews: Can damage reputation
    
    4. Missed Intervention Opportunity:
       - We could have offered retention incentives
       - Typical retention cost: $50-$100
       - Success rate: 30-40% if caught early
    
    COST COMPARISON:
    ---------------
    False Negative cost: $1,200+ (lost customer value)
    False Positive cost: $50-100 (unnecessary retention offer)
    
    Ratio: FN is 12-25x more expensive than FP
    
    CONCLUSION:
    -----------
    We should optimize for RECALL (catching more churners) even if it
    means more False Positives. It's better to give an unnecessary
    discount than lose a customer entirely.
    """
    return explanation


if __name__ == '__main__':
    # Test model training
    from data_generator import generate_telco_dataset
    from preprocessing import ChurnDataProcessor, prepare_train_test_split
    
    # Generate data
    df = generate_telco_dataset(n_samples=2000)
    
    # Preprocess
    processor = ChurnDataProcessor()
    X, y, features = processor.fit_transform(df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # Train models
    trainer = ChurnModelTrainer()
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    
    # Evaluate
    trainer.evaluate_model(
        trainer.models['logistic_regression'],
        X_test, y_test, 'logistic_regression'
    )
    trainer.evaluate_model(
        trainer.models['random_forest'],
        X_test, y_test, 'random_forest'
    )
    
    # Compare
    trainer.compare_models()
    
    # Feature importance
    print("\nTop Features (Random Forest):")
    importance = trainer.get_feature_importance(
        trainer.models['random_forest'], features
    )
    print(importance.to_string())
    
    # Business impact
    y_pred = trainer.best_model.predict(X_test)
    impact = calculate_business_impact(y_test, y_pred)
    print(f"\nBusiness Impact:")
    for k, v in impact.items():
        if isinstance(v, float):
            print(f"  {k}: ${v:,.2f}" if 'cost' in k or 'revenue' in k or 'value' in k or 'loss' in k or 'impact' in k else f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
