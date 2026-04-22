import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Iris Flower Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please make sure 'iris_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()

# Species information
SPECIES_INFO = {
    'Setosa': {
        'description': 'The smallest iris species with characteristic wide petals.',
        'color': '#FF6B6B',
        'icon': '🌸'
    },
    'Versicolor': {
        'description': 'Medium-sized iris with blue-purple flowers.',
        'color': '#4ECDC4',
        'icon': '🌺'
    },
    'Virginica': {
        'description': 'The largest iris species with deep purple flowers.',
        'color': '#45B7D1',
        'icon': '🌷'
    }
}

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌸 Iris Flower Species Predictor</h1>
            <p>Machine Learning Model for Iris Classification using Random Forest</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Input Parameters")
        st.markdown("Adjust the sliders below to input the flower measurements:")
        
        # Create input sliders with ranges based on actual data
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.8,
            step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=4.0,
            step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=1.3,
            step=0.1,
            help="Width of the petal in centimeters"
        )
        
        # Display input values in a nice format
        st.markdown("### 📝 Input Summary")
        input_data_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_data_df, use_container_width=True, hide_index=True)
    
    # Prepare input for prediction
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Map prediction to species name
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction]
    confidence = prediction_proba[prediction] * 100
    
    with col2:
        st.markdown("### 🔮 Prediction Result")
        
        # Display prediction with animation
        st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, {SPECIES_INFO[predicted_species]['color']}20, {SPECIES_INFO[predicted_species]['color']}40);">
                <h2 style="font-size: 3rem; margin: 0;">{SPECIES_INFO[predicted_species]['icon']} {predicted_species}</h2>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">{SPECIES_INFO[predicted_species]['description']}</p>
                <div style="margin-top: 1rem;">
                    <p style="font-size: 1.5rem; margin: 0;">Confidence: {confidence:.1f}%</p>
                    <progress value="{confidence}" max="100" style="width: 100%; height: 20px; border-radius: 10px;"></progress>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display probability distribution
        st.markdown("### 📊 Probability Distribution")
        proba_df = pd.DataFrame({
            'Species': species_names,
            'Probability (%)': prediction_proba * 100
        })
        st.bar_chart(proba_df.set_index('Species'))
    
    # Additional information section
    st.markdown("---")
    st.markdown("### ℹ️ About the Model")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
            <div class="info-box">
                <h4>🎯 Model Type</h4>
                <p>Random Forest Classifier</p>
                <p><small>Ensemble learning method for classification</small></p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="info-box">
                <h4>📈 Model Accuracy</h4>
                <p>~97-100% on test data</p>
                <p><small>Trained on 150 samples from Iris dataset</small></p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
            <div class="info-box">
                <h4>🔧 Features Used</h4>
                <p>• Sepal Length (cm)<br>
                • Sepal Width (cm)<br>
                • Petal Length (cm)<br>
                • Petal Width (cm)</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Built with ❤️ using Streamlit & Scikit-learn | Iris Dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
