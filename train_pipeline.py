"""
Churn Prediction Training Pipeline
===================================
End-to-end script to train and evaluate churn prediction models.

Run this script to:
1. Generate/load training data
2. Preprocess and engineer features
3. Train multiple models
4. Evaluate and compare performance
5. Save best model for deployment

Usage:
    python train_pipeline.py
    python train_pipeline.py --data path/to/your/data.csv
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_telco_dataset
from preprocessing import (
    ChurnDataProcessor, 
    prepare_train_test_split,
    apply_smote,
    get_class_weights
)
from model_training import (
    ChurnModelTrainer,
    calculate_business_impact,
    explain_false_negatives
)


def run_pipeline(data_path=None, output_dir='models', tune_hyperparams=False):
    """
    Execute the complete training pipeline.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to CSV file with customer data. If None, generates synthetic data.
    output_dir : str
        Directory to save trained models
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning (slower but better results)
    """
    print("="*70)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load or Generate Data
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    if data_path and os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic Telco customer data...")
        df = generate_telco_dataset(n_samples=7043, random_state=42)
        
        # Save generated data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(os.path.join(data_dir, 'telco_churn_data.csv'), index=False)
        print(f"Generated and saved {len(df)} customer records")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    
    # =========================================================================
    # STEP 2: Preprocess Data
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    processor = ChurnDataProcessor()
    X, y, feature_names = processor.fit_transform(df)
    
    print(f"\nProcessed features: {len(feature_names)}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Handle class imbalance
    print("\nHandling class imbalance...")
    class_weights = get_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Try SMOTE if available
    try:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
        use_smote = True
    except:
        X_train_balanced, y_train_balanced = X_train, y_train
        use_smote = False
        print("SMOTE not available, using class weights instead")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
    processor.save(preprocessor_path)
    
    # =========================================================================
    # STEP 3: Train Models
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    trainer = ChurnModelTrainer(random_state=42)
    
    # Train Logistic Regression (baseline)
    trainer.train_logistic_regression(
        X_train_balanced if use_smote else X_train,
        y_train_balanced if use_smote else y_train,
        class_weight='balanced' if not use_smote else None
    )
    
    # Train Random Forest
    trainer.train_random_forest(
        X_train_balanced if use_smote else X_train,
        y_train_balanced if use_smote else y_train,
        class_weight='balanced' if not use_smote else None,
        tune_hyperparams=tune_hyperparams
    )
    
    # Try XGBoost if available
    trainer.train_xgboost(
        X_train_balanced if use_smote else X_train,
        y_train_balanced if use_smote else y_train
    )
    
    # =========================================================================
    # STEP 4: Evaluate Models
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    for name, model in trainer.models.items():
        trainer.evaluate_model(model, X_test, y_test, name)
    
    # Compare all models
    comparison = trainer.compare_models()
    
    # Find optimal threshold for best model
    print("\n" + "-"*50)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("-"*50)
    optimal_threshold, threshold_results = trainer.find_optimal_threshold(
        trainer.best_model, X_test, y_test,
        fn_cost=500, fp_cost=50
    )
    
    # =========================================================================
    # STEP 5: Feature Importance
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: FEATURE IMPORTANCE")
    print("="*70)
    
    importance_df = trainer.get_feature_importance(
        trainer.best_model, feature_names, top_n=15
    )
    print("\nTop 15 Most Important Features:")
    print(importance_df.to_string())
    
    # Save feature importance
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # =========================================================================
    # STEP 6: Business Impact Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: BUSINESS IMPACT ANALYSIS")
    print("="*70)
    
    y_pred = trainer.best_model.predict(X_test)
    y_proba = trainer.best_model.predict_proba(X_test)[:, 1]
    
    # Using optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    print("\nWith default threshold (0.5):")
    impact_default = calculate_business_impact(y_test, y_pred)
    for k, v in impact_default.items():
        if isinstance(v, float) and ('cost' in k or 'revenue' in k or 'value' in k or 'loss' in k or 'impact' in k):
            print(f"  {k}: ${v:,.0f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nWith optimal threshold ({optimal_threshold:.2f}):")
    impact_optimal = calculate_business_impact(y_test, y_pred_optimal)
    for k, v in impact_optimal.items():
        if isinstance(v, float) and ('cost' in k or 'revenue' in k or 'value' in k or 'loss' in k or 'impact' in k):
            print(f"  {k}: ${v:,.0f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # =========================================================================
    # STEP 7: Save Models and Results
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING ARTIFACTS")
    print("="*70)
    
    # Save best model
    model_path = os.path.join(output_dir, 'churn_model.joblib')
    trainer.save_model(trainer.best_model, model_path)
    
    # Save all models
    for name, model in trainer.models.items():
        path = os.path.join(output_dir, f'{name}_model.joblib')
        trainer.save_model(model, path)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    trainer.save_results(results_path)
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Save pipeline metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(feature_names),
        'best_model': trainer.best_model_name,
        'optimal_threshold': optimal_threshold,
        'use_smote': use_smote,
        'metrics': trainer.results[trainer.best_model_name]
    }
    with open(os.path.join(output_dir, 'pipeline_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nAll artifacts saved to: {output_dir}/")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"""
    Best Model: {trainer.best_model_name}
    
    Performance Metrics:
    - Accuracy:  {trainer.results[trainer.best_model_name]['accuracy']:.3f}
    - Precision: {trainer.results[trainer.best_model_name]['precision']:.3f}
    - Recall:    {trainer.results[trainer.best_model_name]['recall']:.3f}
    - F1-Score:  {trainer.results[trainer.best_model_name]['f1_score']:.3f}
    - ROC-AUC:   {trainer.results[trainer.best_model_name]['roc_auc']:.3f}
    
    Optimal Threshold: {optimal_threshold:.2f}
    
    Business Impact (per {len(y_test)} customers):
    - Estimated model value: ${impact_optimal['model_value']:,.0f}
    - ROI: {impact_optimal['roi_percentage']:.0f}%
    
    Files saved:
    - {model_path}
    - {preprocessor_path}
    - {results_path}
    - {output_dir}/feature_importance.csv
    - {output_dir}/pipeline_metadata.json
    
    Next Steps:
    1. Run the Streamlit app: streamlit run app.py
    2. Review feature importance for business insights
    3. Set up model monitoring for production
    """)
    
    return trainer, processor, feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train churn prediction models')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to customer data CSV (optional)')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        tune_hyperparams=args.tune
    )
