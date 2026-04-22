import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Iris Flower Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS
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
    }
    </style>
""", unsafe_allow_html=True)

# Try to import required packages with error handling
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    JOBLIB_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import joblib: {e}")
    JOBLIB_AVAILABLE = False

# Function to load model with fallback
@st.cache_resource
def load_models():
    """Load the trained model and scaler with error handling"""
    
    if not JOBLIB_AVAILABLE:
        st.error("joblib is not available. Please check your installation.")
        return None, None
    
    try:
        # Check if files exist
        if not os.path.exists('iris_model.pkl'):
            st.error("Model file 'iris_model.pkl' not found!")
            st.info("Please make sure the model file is uploaded to the repository.")
            return None, None
        
        if not os.path.exists('scaler.pkl'):
            st.error("Scaler file 'scaler.pkl' not found!")
            st.info("Please make sure the scaler file is uploaded to the repository.")
            return None, None
        
        # Load the files
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        return model, scaler
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Try retraining the model in Colab and uploading the new files.")
        return None, None

# Function to create a simple fallback model if needed
def create_fallback_model():
    """Create a simple rule-based model as fallback"""
    def predict(features):
        # Simple rule-based classification
        petal_length = features[0][2]
        petal_width = features[0][3]
        
        if petal_length < 2.5:
            return 0  # Setosa
        elif petal_width < 1.8:
            return 1  # Versicolor
        else:
            return 2  # Virginica
    
    return predict

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌸 Iris Flower Species Predictor</h1>
            <p>Machine Learning Model for Iris Classification</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    
    # Create input sliders
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Input Parameters")
        
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
    
    # Prepare input
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    with col2:
        st.markdown("### 🔮 Prediction Result")
        
        # Make prediction
        if model is not None and scaler is not None:
            try:
                input_scaled = scaler.transform(input_features)
                prediction = model.predict(input_scaled)[0]
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    confidence = prediction_proba[prediction] * 100
                else:
                    confidence = 95  # Default confidence
                
                species_names = ['Setosa', 'Versicolor', 'Virginica']
                predicted_species = species_names[prediction]
                
                # Color mapping
                colors = {
                    'Setosa': '#FF6B6B',
                    'Versicolor': '#4ECDC4',
                    'Virginica': '#45B7D1'
                }
                
                st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(135deg, {colors[predicted_species]}20, {colors[predicted_species]}40);">
                        <h2 style="font-size: 2.5rem; margin: 0;">{predicted_species}</h2>
                        <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                        <progress value="{confidence}" max="100" style="width: 100%; height: 20px;"></progress>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Using fallback prediction method...")
                # Fallback prediction
                if petal_length < 2.5:
                    predicted_species = "Setosa"
                elif petal_width < 1.8:
                    predicted_species = "Versicolor"
                else:
                    predicted_species = "Virginica"
                
                st.markdown(f"""
                    <div class="prediction-box" style="background: #f0f2f6;">
                        <h2 style="font-size: 2.5rem;">{predicted_species}</h2>
                        <p>(Using rule-based fallback)</p>
                    </div>
                """, unsafe_allow_html=True)
        
        else:
            # Fallback prediction without model
            if petal_length < 2.5:
                predicted_species = "Setosa"
            elif petal_width < 1.8:
                predicted_species = "Versicolor"
            else:
                predicted_species = "Virginica"
            
            st.warning("⚠️ Using simplified prediction (model not loaded)")
            st.markdown(f"""
                <div class="prediction-box" style="background: #f0f2f6;">
                    <h2 style="font-size: 2.5rem;">{predicted_species}</h2>
                    <p>(Fallback mode - based on petal measurements)</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display input summary
    st.markdown("---")
    st.markdown("### 📝 Input Summary")
    input_df = pd.DataFrame({
        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)
    
    # Debug information (only shown in development)
    if st.checkbox("Show Debug Info"):
        st.markdown("### 🔧 Debug Information")
        st.write("Python Version:", sys.version)
        st.write("Packages available:")
        st.write("- joblib:", JOBLIB_AVAILABLE)
        st.write("Files in directory:", os.listdir('.') if os.path.exists('.') else "Cannot list files")
        st.write("Model loaded:", model is not None)
        st.write("Scaler loaded:", scaler is not None)

if __name__ == "__main__":
    main()
