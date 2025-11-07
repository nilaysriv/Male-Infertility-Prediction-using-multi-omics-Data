import pickle
import os
import pandas as pd

# --- Load Model Artifacts ---
# This is done once when the module is loaded.
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'infertility_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    MODEL = artifacts['model']
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please run the model training script.")
    MODEL = None


def predict(processed_data):
    """
    Makes a prediction on preprocessed data using the loaded model.
    """
    if MODEL is None:
        return {"error": "Model is not loaded."}
    
    # --- Prediction ---
    # Predict the class (0 for Fertile, 1 for Infertile) and the probability for each class.
    predictions = MODEL.predict(processed_data)
    probabilities = MODEL.predict_proba(processed_data)
    
    # --- Feature Importance ---
    # Extract the feature importances from the RandomForest model.
    importances = MODEL.feature_importances_
    feature_names = processed_data.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # --- Format Results ---
    # Since we predict on a per-sample basis, we average the results for a single study-level prediction.
    avg_prediction = "Infertile" if predictions.mean() > 0.5 else "Fertile"
    avg_confidence = probabilities[:, 1].mean() # Average confidence for the "Infertile" class

    result = {
        'prediction': avg_prediction,
        'confidence': f"{avg_confidence:.2%}",
        'feature_importances': feature_importance_df.to_dict('records')
    }
    
    return result

