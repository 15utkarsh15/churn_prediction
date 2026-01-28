"""
Telco Customer Churn Data Generator
Creates realistic synthetic data matching the IBM Telco Customer Churn dataset structure.
This script generates data that mimics real-world churn patterns based on known factors.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import uuid


def generate_customer_id() -> str:
    """Generate a customer ID in the format XXXX-XXXXX"""
    part1 = ''.join(np.random.choice(list('0123456789'), 4))
    part2 = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))
    return f"{part1}-{part2}"


def generate_telco_dataset(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Telco Customer Churn dataset.
    
    The data generation incorporates real-world churn drivers:
    - Short tenure customers churn more
    - Month-to-month contracts have higher churn
    - Fiber optic customers without security services churn more
    - Electronic check payment has higher churn
    - Senior citizens have slightly higher churn
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Complete dataset matching Telco churn schema
    """
    np.random.seed(random_state)
    
    # Generate base customer attributes
    customer_ids = [generate_customer_id() for _ in range(n_samples)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Tenure (months) - bimodal distribution common in subscription services
    # Mix of new customers and long-term loyal customers
    n_new = int(n_samples * 0.4)
    n_loyal = n_samples - n_new
    tenure_new = np.random.exponential(scale=8, size=n_new)
    tenure_loyal = np.random.normal(loc=55, scale=15, size=n_loyal)
    tenure = np.concatenate([tenure_new, tenure_loyal])
    np.random.shuffle(tenure)
    tenure = np.clip(tenure, 0, 72).astype(int)
    
    # Phone Service
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple Lines (depends on phone service)
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))
    multiple_lines = np.array(multiple_lines)
    
    # Internet Service
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        n_samples, 
        p=[0.34, 0.44, 0.22]
    )
    
    # Internet-dependent services
    internet_dependent_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                               'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    internet_services = {}
    for col in internet_dependent_cols:
        service_vals = []
        for inet in internet_service:
            if inet == 'No':
                service_vals.append('No internet service')
            else:
                # Security and support services less common with Fiber optic (realistic pattern)
                if col in ['OnlineSecurity', 'TechSupport'] and inet == 'Fiber optic':
                    service_vals.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
                else:
                    service_vals.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
        internet_services[col] = np.array(service_vals)
    
    # Contract Type
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.55, 0.21, 0.24]
    )
    
    # Paperless Billing
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    
    # Payment Method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Monthly Charges (based on services)
    monthly_charges = []
    for i in range(n_samples):
        base_charge = 18.0  # Base phone charge
        
        if phone_service[i] == 'Yes':
            base_charge += 2.0
            if multiple_lines[i] == 'Yes':
                base_charge += 5.0
        
        if internet_service[i] == 'DSL':
            base_charge += 25.0
        elif internet_service[i] == 'Fiber optic':
            base_charge += 45.0
        
        for col in internet_dependent_cols:
            if internet_services[col][i] == 'Yes':
                base_charge += np.random.uniform(8, 12)
        
        # Add some noise
        base_charge += np.random.normal(0, 3)
        monthly_charges.append(max(18.0, round(base_charge, 2)))
    
    monthly_charges = np.array(monthly_charges)
    
    # Total Charges
    total_charges = []
    for i in range(n_samples):
        if tenure[i] == 0:
            total_charges.append(monthly_charges[i])
        else:
            # Some variation in historical charges
            avg_historical = monthly_charges[i] * 0.95  # Slight discount assumption
            tc = tenure[i] * avg_historical + np.random.normal(0, 50)
            total_charges.append(max(monthly_charges[i], round(tc, 2)))
    
    total_charges = np.array(total_charges)
    
    # Generate Churn based on realistic factors
    churn_probs = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.15  # Base churn probability
        
        # Tenure effect (biggest driver)
        if tenure[i] <= 6:
            prob += 0.25
        elif tenure[i] <= 12:
            prob += 0.15
        elif tenure[i] <= 24:
            prob += 0.05
        elif tenure[i] >= 48:
            prob -= 0.08
        
        # Contract effect
        if contract[i] == 'Month-to-month':
            prob += 0.18
        elif contract[i] == 'One year':
            prob -= 0.05
        else:  # Two year
            prob -= 0.12
        
        # Internet service effect
        if internet_service[i] == 'Fiber optic':
            prob += 0.08
            # Lack of security services with fiber
            if internet_services['OnlineSecurity'][i] == 'No':
                prob += 0.05
            if internet_services['TechSupport'][i] == 'No':
                prob += 0.05
        
        # Payment method effect
        if payment_method[i] == 'Electronic check':
            prob += 0.10
        elif 'automatic' in payment_method[i]:
            prob -= 0.05
        
        # Senior citizen effect
        if senior_citizen[i] == 1:
            prob += 0.05
        
        # Partner/dependents effect (social ties reduce churn)
        if partner[i] == 'Yes':
            prob -= 0.03
        if dependents[i] == 'Yes':
            prob -= 0.03
        
        # High monthly charges without long-term contract
        if monthly_charges[i] > 80 and contract[i] == 'Month-to-month':
            prob += 0.08
        
        # Paperless billing slight effect
        if paperless_billing[i] == 'Yes':
            prob += 0.02
        
        churn_probs[i] = np.clip(prob, 0.02, 0.85)
    
    # Generate actual churn based on probabilities
    churn = np.array(['Yes' if np.random.random() < p else 'No' for p in churn_probs])
    
    # Build DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': internet_services['OnlineSecurity'],
        'OnlineBackup': internet_services['OnlineBackup'],
        'DeviceProtection': internet_services['DeviceProtection'],
        'TechSupport': internet_services['TechSupport'],
        'StreamingTV': internet_services['StreamingTV'],
        'StreamingMovies': internet_services['StreamingMovies'],
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    # Introduce some realistic missing values in TotalCharges (like original dataset)
    # New customers sometimes have blank TotalCharges
    new_customer_mask = df['tenure'] == 0
    blank_indices = df[new_customer_mask].sample(frac=0.5, random_state=random_state).index
    df.loc[blank_indices, 'TotalCharges'] = ' '
    
    return df


def get_feature_descriptions() -> dict:
    """Return descriptions of each feature for documentation."""
    return {
        'customerID': 'Unique identifier for each customer',
        'gender': 'Customer gender (Male/Female)',
        'SeniorCitizen': 'Whether customer is senior citizen (1=Yes, 0=No)',
        'Partner': 'Whether customer has a partner (Yes/No)',
        'Dependents': 'Whether customer has dependents (Yes/No)',
        'tenure': 'Number of months customer has been with company',
        'PhoneService': 'Whether customer has phone service (Yes/No)',
        'MultipleLines': 'Whether customer has multiple lines (Yes/No/No phone service)',
        'InternetService': 'Type of internet service (DSL/Fiber optic/No)',
        'OnlineSecurity': 'Whether customer has online security add-on (Yes/No/No internet service)',
        'OnlineBackup': 'Whether customer has online backup add-on (Yes/No/No internet service)',
        'DeviceProtection': 'Whether customer has device protection add-on (Yes/No/No internet service)',
        'TechSupport': 'Whether customer has tech support add-on (Yes/No/No internet service)',
        'StreamingTV': 'Whether customer has streaming TV add-on (Yes/No/No internet service)',
        'StreamingMovies': 'Whether customer has streaming movies add-on (Yes/No/No internet service)',
        'Contract': 'Contract term (Month-to-month/One year/Two year)',
        'PaperlessBilling': 'Whether customer uses paperless billing (Yes/No)',
        'PaymentMethod': 'Payment method (Electronic check/Mailed check/Bank transfer/Credit card)',
        'MonthlyCharges': 'Monthly charge amount in dollars',
        'TotalCharges': 'Total charges to date in dollars',
        'Churn': 'Whether customer churned (Yes/No) - TARGET VARIABLE'
    }


if __name__ == '__main__':
    # Generate and save dataset
    print("Generating Telco Customer Churn dataset...")
    df = generate_telco_dataset(n_samples=7043)
    
    # Save to CSV
    output_path = '../data/telco_churn_generated.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nChurn distribution:")
    print(df['Churn'].value_counts(normalize=True))
    print(f"\nSample records:")
    print(df.head())
