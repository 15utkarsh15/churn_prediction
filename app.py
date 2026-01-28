"""
Customer Churn Prediction Dashboard
====================================
Streamlit application for predicting and analyzing customer churn risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal clean CSS
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    h1 { color: #1e3a5f; font-weight: 600; }
    h2, h3 { color: #2c5282; }
    .stButton > button {
        background-color: #3182ce;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stButton > button:hover { background-color: #2c5282; }
</style>
""", unsafe_allow_html=True)


def load_models():
    """Load trained models and preprocessor."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'churn_model.joblib')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        preprocessor_state = joblib.load(preprocessor_path)
        
        # Reconstruct the preprocessor from saved state
        from src.preprocessing import ChurnDataProcessor
        preprocessor = ChurnDataProcessor()
        preprocessor.label_encoders = preprocessor_state['label_encoders']
        preprocessor.scaler = preprocessor_state['scaler']
        preprocessor.feature_names = preprocessor_state['feature_names']
        preprocessor.categorical_cols = preprocessor_state['categorical_cols']
        preprocessor.numerical_cols = preprocessor_state['numerical_cols']
        preprocessor.binary_cols = preprocessor_state['binary_cols']
        preprocessor._fitted = preprocessor_state['_fitted']
        
        return model, preprocessor
    return None, None


def get_risk_color(prob):
    if prob >= 0.7: return "#e53e3e"
    elif prob >= 0.4: return "#ed8936"
    return "#48bb78"


def get_risk_level(prob):
    if prob >= 0.7: return "HIGH RISK"
    elif prob >= 0.4: return "MEDIUM RISK"
    return "LOW RISK"


def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk Score", 'font': {'size': 18, 'color': '#1e3a5f'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#1e3a5f'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': get_risk_color(probability)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#c6f6d5'},
                {'range': [40, 70], 'color': '#feebc8'},
                {'range': [70, 100], 'color': '#fed7d7'}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_feature_impact_chart(contributions, top_n=8):
    df = contributions.copy()
    df['abs_impact'] = df['shap_value'].abs()
    top = df.nlargest(top_n, 'abs_impact')
    colors = ['#e53e3e' if x > 0 else '#48bb78' for x in top['shap_value']]
    
    fig = go.Figure(go.Bar(
        y=top['feature'], x=top['shap_value'], orientation='h',
        marker_color=colors,
        text=[f"+{x:.2f}" if x > 0 else f"{x:.2f}" for x in top['shap_value']],
        textposition='outside'
    ))
    fig.update_layout(
        title='Top Factors Affecting This Customer',
        xaxis_title='Impact on Churn Risk', yaxis_title='',
        height=320, margin=dict(l=20, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(zeroline=True, zerolinecolor='#718096'),
        yaxis=dict(autorange='reversed')
    )
    return fig


def sidebar_customer_input():
    """Create sidebar form for customer data input."""
    st.sidebar.header("📋 Customer Information")
    st.sidebar.markdown("---")
    
    # Demographics
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    
    # Account
    st.sidebar.markdown("---")
    st.sidebar.subheader("Account Details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", 
                                    ["Month-to-month", "One year", "Two year"])
    
    # Services
    st.sidebar.markdown("---")
    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = "No phone service" if phone_service == "No" else \
                     st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
    
    internet_service = st.sidebar.selectbox("Internet Service", 
                                            ["Fiber optic", "DSL", "No"])
    
    if internet_service != "No":
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
    
    # Billing
    st.sidebar.markdown("---")
    st.sidebar.subheader("Billing")
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
         "Credit card (automatic)"])
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = monthly_charges * max(tenure, 1)
    st.sidebar.text(f"Est. Total Charges: ${total_charges:,.2f}")
    
    return {
        'customerID': 'NEW-CUSTOMER', 'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'Churn': 'No'
    }


def simulate_prediction(customer_data):
    """Simulate prediction based on known churn factors."""
    risk = 0.15
    if customer_data['Contract'] == 'Month-to-month': risk += 0.22
    if customer_data['tenure'] < 6: risk += 0.18
    elif customer_data['tenure'] < 12: risk += 0.10
    if customer_data['InternetService'] == 'Fiber optic': risk += 0.08
    if customer_data['PaymentMethod'] == 'Electronic check': risk += 0.10
    if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No': 
        risk += 0.05
    if customer_data['TechSupport'] == 'No' and customer_data['InternetService'] != 'No': 
        risk += 0.05
    if customer_data['SeniorCitizen'] == 1: risk += 0.04
    if customer_data['Partner'] == 'Yes': risk -= 0.03
    if customer_data['Dependents'] == 'Yes': risk -= 0.03
    return min(max(risk, 0.05), 0.92)


def get_simulated_contributions(customer_data, prob):
    """Generate simulated feature contributions."""
    return pd.DataFrame({
        'feature': ['Contract Type', 'Tenure', 'Internet Service', 'Payment Method',
                   'Online Security', 'Tech Support', 'Monthly Charges', 'Family Status'],
        'shap_value': [
            0.18 if customer_data['Contract'] == 'Month-to-month' else -0.08,
            max(-0.15, 0.20 - 0.005 * customer_data['tenure']),
            0.08 if customer_data['InternetService'] == 'Fiber optic' else -0.03,
            0.10 if customer_data['PaymentMethod'] == 'Electronic check' else -0.04,
            0.05 if customer_data['OnlineSecurity'] == 'No' else -0.05,
            0.04 if customer_data['TechSupport'] == 'No' else -0.04,
            0.002 * (customer_data['MonthlyCharges'] - 60),
            -0.05 if customer_data['Partner'] == 'Yes' else 0.02
        ]
    })


def show_recommendations(prob, data):
    """Display actionable recommendations."""
    st.markdown("### 📋 Recommended Actions")
    
    if prob >= 0.7:
        st.error("**Immediate Action Required**")
        recs = [
            "Contact customer within 24 hours via retention team",
            "Prepare 15-20% discount or service upgrade offer",
            "Review any recent support tickets or complaints",
            "Offer incentive to switch to annual contract"
        ]
    elif prob >= 0.4:
        st.warning("**Proactive Engagement Recommended**")
        recs = [
            "Send personalized satisfaction survey",
            "Consider loyalty rewards or small discount",
            "Check if customer is using all available features",
            "Offer service bundle if on single service"
        ]
    else:
        st.success("**Customer Appears Stable**")
        recs = [
            "Continue regular engagement",
            "Consider upsell opportunities",
            "Invite to referral program",
            "Request testimonial or review"
        ]
    
    for i, r in enumerate(recs, 1):
        st.markdown(f"{i}. {r}")


def main():
    st.title("📊 Customer Churn Prediction")
    st.markdown("*Identify at-risk customers and take action before they leave*")
    st.markdown("---")
    
    model, preprocessor = load_models()
    demo_mode = model is None
    
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📈 Analytics", "📚 About"])
    
    with tab1:
        if demo_mode:
            st.info("**Demo Mode**: Models not loaded. Using rule-based simulation. "
                   "Run `python train_pipeline.py` to train actual models.")
        
        customer_data = sidebar_customer_input()
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔮 Predict Churn Risk", use_container_width=True):
            with st.spinner("Analyzing customer profile..."):
                if demo_mode:
                    prob = simulate_prediction(customer_data)
                    contributions = get_simulated_contributions(customer_data, prob)
                else:
                    df = pd.DataFrame([customer_data])
                    X = preprocessor.transform(df)
                    prob = model.predict_proba(X)[0, 1]
                    contributions = None  # Would use SHAP here
            
            # Results display
            risk_color = get_risk_color(prob)
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f"""
                <div style='background:{risk_color}; padding:1.2rem; border-radius:10px; 
                            text-align:center; color:white;'>
                    <h3 style='margin:0; color:white;'>{get_risk_level(prob)}</h3>
                    <p style='font-size:2.2rem; margin:0.3rem 0; font-weight:bold;'>{prob:.1%}</p>
                    <p style='margin:0; opacity:0.9;'>Churn Probability</p>
                </div>""", unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div style='background:#f7fafc; padding:1.2rem; border-radius:10px; 
                            border:1px solid #e2e8f0; text-align:center;'>
                    <h4 style='margin:0; color:#2d3748;'>Tenure</h4>
                    <p style='font-size:1.8rem; margin:0.3rem 0; color:#1e3a5f; font-weight:bold;'>
                        {customer_data['tenure']} mo</p>
                    <p style='margin:0; color:#718096;'>{customer_data['Contract']}</p>
                </div>""", unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div style='background:#f7fafc; padding:1.2rem; border-radius:10px; 
                            border:1px solid #e2e8f0; text-align:center;'>
                    <h4 style='margin:0; color:#2d3748;'>Monthly Value</h4>
                    <p style='font-size:1.8rem; margin:0.3rem 0; color:#1e3a5f; font-weight:bold;'>
                        ${customer_data['MonthlyCharges']:.0f}</p>
                    <p style='margin:0; color:#718096;'>{customer_data['InternetService']}</p>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
            with col2:
                if contributions is not None:
                    st.plotly_chart(create_feature_impact_chart(contributions), 
                                   use_container_width=True)
            
            st.markdown("---")
            show_recommendations(prob, customer_data)
        else:
            st.info("👈 Enter customer details in the sidebar and click **Predict Churn Risk**")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Churn Rate", "26.5%", help="Telecom industry average")
            c2.metric("Customer Value", "$1,200/yr", help="Average annual revenue")
            c3.metric("Retention Success", "30%", help="Typical intervention success")
    
    with tab2:
        st.header("📈 Churn Analytics")
        
        c1, c2 = st.columns(2)
        
        with c1:
            contract_df = pd.DataFrame({
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'Churn Rate': [42.7, 11.3, 2.8]
            })
            fig = px.bar(contract_df, x='Contract', y='Churn Rate',
                        title='Churn Rate by Contract Type',
                        color='Churn Rate', color_continuous_scale='RdYlGn_r')
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            tenure_df = pd.DataFrame({
                'Tenure': ['0-6 mo', '6-12 mo', '1-2 yr', '2-4 yr', '4+ yr'],
                'Churn Rate': [47.8, 28.2, 21.4, 15.1, 8.7]
            })
            fig = px.bar(tenure_df, x='Tenure', y='Churn Rate',
                        title='Churn Rate by Tenure',
                        color='Churn Rate', color_continuous_scale='RdYlGn_r')
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Key Insights")
        
        insights = """
        | Factor | Impact | Business Recommendation |
        |--------|--------|------------------------|
        | Month-to-month contracts | +43% churn | Incentivize annual contracts |
        | Tenure < 6 months | +48% churn | Enhanced onboarding program |
        | Fiber optic (no security) | +35% churn | Bundle security services |
        | Electronic check payment | +32% churn | Promote auto-pay discounts |
        | No tech support | +28% churn | Offer support add-on deals |
        """
        st.markdown(insights)
    
    with tab3:
        st.header("📚 About This Tool")
        
        st.markdown("""
        ### Purpose
        This tool predicts the likelihood of customer churn using machine learning,
        enabling proactive retention efforts before customers leave.
        
        ### How It Works
        1. **Input**: Customer demographics, services, and billing information
        2. **Analysis**: ML model calculates churn probability based on historical patterns
        3. **Output**: Risk score with explanations and recommended actions
        
        ### Model Details
        - **Algorithms**: Logistic Regression (baseline), Random Forest (primary)
        - **Key Metrics**: Optimized for Recall (catching churners) while maintaining Precision
        - **Accuracy**: ~80% overall, ~75% recall on churners
        
        ### Business Impact
        
        **Cost of Errors:**
        - **False Negative** (missed churner): ~$1,200 lost revenue
        - **False Positive** (unnecessary retention offer): ~$50 cost
        - **Ratio**: Missing a churner costs 24x more than a false alarm
        
        **Expected ROI:**
        - Identifying 100 at-risk customers → ~30 saved with intervention
        - Revenue saved: 30 × $1,200 = $36,000
        - Retention cost: 100 × $50 = $5,000
        - **Net benefit: $31,000 per 100 predictions**
        
        ### Data Privacy
        All customer data is processed locally. No data is stored or transmitted externally.
        """)
        
        st.markdown("---")
        st.markdown("*Built for business stakeholders*")


if __name__ == "__main__":
    main()
