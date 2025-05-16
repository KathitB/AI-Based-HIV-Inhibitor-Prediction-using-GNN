import os
import numpy as np
import pandas as pd
import deepchem as dc
import streamlit as st
from deepchem.data import NumpyDataset
from deepchem.feat import MolGraphConvFeaturizer
from src.model import build_gnn_model
from src.preprocess import feature_data 
from src.utils import set_seed
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="AI Drug Discovery", layout="wide")
st.title("🧪 AI-Based Drug Discovery (HIV Inhibitor Prediction)")

# Set random seed
set_seed(42)

# Load trained model
@st.cache_resource
def load_trained_model():
    model_dir = "C:/Internship/DRUG_DISCOVERY_HIV_INHIBITOR/outputs/model"
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir=model_dir)

    if os.path.exists(os.path.join(model_dir, "checkpoint")):
        model.restore()
        return model
    else:
        st.warning("⚠️ No trained model checkpoint found. Please train the model first.")
        return None

model = load_trained_model()
if model is None:
    st.stop()  # Stop app if model is not loaded

# Predict for a single SMILES string
def predict_single_smiles(smiles_str):
    featurizer = dc.feat.ConvMolFeaturizer()
    try:
        mol = featurizer.featurize([smiles_str])
        dataset = NumpyDataset(mol)
        prediction = model.predict(dataset)*100
        return prediction[0][0]
    except Exception as e:
        st.error(f"❌ Featurization or prediction failed: {e}")
        return None

# Predict from uploaded CSV file
def predict_from_csv(file):
    file.seek(0)
    df = pd.read_csv(file)

    if 'smiles' not in df.columns:
        st.error("CSV must contain a 'smiles' column")
        return None

    smiles_list = df['smiles'].tolist()
    featurizer = dc.feat.ConvMolFeaturizer()

    features = []
    valid_smiles = []
    for smi in smiles_list:
        try:
            feat = featurizer.featurize([smi])[0]
            if feat is not None:
                features.append(feat)
                valid_smiles.append(smi)
        except Exception as e:
            print(f"Failed to featurize {smi}: {e}")
            continue

    if len(features) == 0:
        st.error("❌ No valid SMILES found.")
        return None

    dataset = dc.data.NumpyDataset(X=features)
    predictions = model.predict(dataset)*100

    # If predictions has shape (n,2), take second column (prob class 1)
    if predictions.ndim > 1 and predictions.shape[1] == 2:
        predictions = predictions[:, 1]

    predictions = predictions.flatten()

    print(f"Valid smiles length: {len(valid_smiles)}")
    print(f"Predictions length: {len(predictions)}")

    # Safety check:
    min_len = min(len(valid_smiles), len(predictions))
    if len(valid_smiles) != len(predictions):
        st.warning(f"Length mismatch! Truncating to shortest length {min_len}")

    result_df = pd.DataFrame({
        'smiles': valid_smiles[:min_len],
        'Predicted_HIV_Activity': predictions[:min_len]
    })

    return result_df




# User interface
option = st.radio("Choose Input Type", ["Single SMILES", "CSV Upload"])

if option == "Single SMILES":
    smiles_input = st.text_input("Enter SMILES string", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1")
    if st.button("Predict"):
        pred = predict_single_smiles(smiles_input)
        if pred is not None:
            st.success(f"✅ Predicted HIV Activity Score: {pred[0]:.4f}")


else:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'smiles' column", type=["csv"])
    if uploaded_file is not None:
        st.write("📄 Preview of Uploaded Data:")
        st.dataframe(pd.read_csv(uploaded_file).head())

        if st.button("Predict All"):
            result_df = predict_from_csv(uploaded_file)
            if result_df is not None:
                st.success("✅ Predictions complete!")
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
