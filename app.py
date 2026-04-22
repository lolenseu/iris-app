# app_standalone.py - No external model files needed
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Species Predictor")

# Train model on the fly
@st.cache_resource
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, iris.target_names, accuracy

# Load or train model
model, species_names, accuracy = train_model()

st.success(f"✅ Model loaded! Accuracy: {accuracy*100:.1f}%")

# Input sliders
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

predicted_species = species_names[prediction]
confidence = probabilities[prediction] * 100

# Display result
st.markdown("---")
st.markdown("## 🔮 Prediction Result")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea20, #764ba240); border-radius: 10px;">
        <h1 style="font-size: 3rem;">{predicted_species}</h1>
        <p style="font-size: 1.5rem;">Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    proba_df = pd.DataFrame({
        'Species': species_names,
        'Probability': probabilities
    })
    st.bar_chart(proba_df.set_index('Species'))

# Input summary
st.markdown("---")
st.markdown("### 📊 Input Measurements")
input_summary = pd.DataFrame({
    'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
})
st.dataframe(input_summary, hide_index=True)
