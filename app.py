import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Hide default Streamlit footer & add custom one
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:'✨ Made by Imtiaz ✨'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 10px;
        color: black;
        text-align: center;
        font-size: 16px;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🔮 ANN Model with Streamlit (Scikit-learn)")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview:", data.head())

    # Select target column
    target_col = st.selectbox("Select Target Column", data.columns)

    if st.button("Train Model"):
        # Features & Target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':  
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ANN Model
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

        # Show some predictions
        st.write("🔍 Sample Predictions:", y_pred[:10])