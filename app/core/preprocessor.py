import pandas as pd
import pickle
import os

# --- Load Model Artifacts ---
# This is done once when the module is loaded for efficiency.
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'infertility_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    
    FEATURE_LIST = artifacts['features']
    SCALER = artifacts['scaler']

except FileNotFoundError:
    print(f"ERROR: Model artifacts not found at {MODEL_PATH}. Please run the model training script first.")
    FEATURE_LIST = []
    SCALER = None


def preprocess_prediction_data(raw_data):
    """
    Preprocesses new, unseen data to match the format the model was trained on.
    """
    if SCALER is None or not FEATURE_LIST:
        print("Model artifacts are not loaded. Cannot preprocess data.")
        return None

    print("Preprocessing new data for prediction...")
    # Transpose data so that genes are columns and samples are rows.
    data_transposed = raw_data.T

    # --- Feature Alignment ---
    # Ensure the new data has the exact same columns (genes) as the training data.
    # Add missing columns with a value of 0, and drop columns not seen during training.
    data_aligned = data_transposed.reindex(columns=FEATURE_LIST, fill_value=0)

    # Check for empty dataframes after alignment
    if data_aligned.empty:
        return None
        
    # --- Scaling ---
    # Apply the *same* scaling transformation that was fitted on the training data.
    scaled_data = SCALER.transform(data_aligned)
    
    final_df = pd.DataFrame(scaled_data, index=data_aligned.index, columns=data_aligned.columns)
    print("Prediction data preprocessing complete.")
    return final_df

