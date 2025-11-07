import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import GEOparse
import tempfile
import time
import requests
from tqdm import tqdm

# --- Constants and Configuration ---
N_FEATURES = 50
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'infertility_model.pkl')
TRAINING_ACCESSION_ID = "GSE45885"


def download_gse_with_retries(accession_id, destdir, retries=3, delay=5):
    """
    Downloads a GSE SOFT file with a progress bar and retry mechanism, then parses it.
    """
    series_prefix = accession_id[3:-3]
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{series_prefix}nnn/{accession_id}/soft/"
        f"{accession_id}_family.soft.gz"
    )
    local_filename = os.path.join(destdir, f"{accession_id}_family.soft.gz")

    for attempt in range(retries):
        try:
            print(f"Downloading from {url} (Attempt {attempt + 1}/{retries})...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=accession_id)
                with open(local_filename, 'wb') as f:
                    for data in r.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    raise IOError("ERROR: Downloaded size did not match expected size.")
            
            print("\nDownload complete. Parsing file...")
            gse = GEOparse.get_GEO(filepath=local_filename, silent=True)
            print("Parsing successful.")
            return gse
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if os.path.exists(local_filename): os.remove(local_filename)
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All download attempts failed.")
    return None


def fetch_data(accession_id):
    """
    Downloads and parses a GEO dataset, returning the expression matrix and the GSE object.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gse = download_gse_with_retries(accession_id, destdir=tmpdir)
        if gse is None: return None, None
        
        all_samples_data = [
            gsm.table[['ID_REF', 'VALUE']].rename(columns={'VALUE': gsm_name}).set_index('ID_REF')
            for gsm_name, gsm in gse.gsms.items() if not gsm.table.empty
        ]

        if not all_samples_data:
            print("Error: No data tables found in GEO samples.")
            return None, None

        expression_matrix = pd.concat(all_samples_data, axis=1)
        print("Successfully built expression matrix.")
        return expression_matrix, gse


def preprocess_training_data(raw_data):
    """
    Cleans, selects features, and scales raw data for training.
    Returns the processed data, the list of selected features, and the fitted scaler.
    """
    if raw_data is None or raw_data.empty: return None, None, None
    print("Preprocessing training data...")
    
    data = raw_data.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0)
    data_transposed = data.T
    
    variances = data_transposed.var()
    top_genes = variances.nlargest(N_FEATURES).index
    processed_data = data_transposed[top_genes]
    print(f"Selected top {N_FEATURES} most variant features (genes).")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    
    final_df = pd.DataFrame(scaled_data, index=processed_data.index, columns=processed_data.columns)
    print("Data preprocessing complete.")
    return final_df, top_genes.tolist(), scaler


def create_labels_from_metadata(gse, sample_order):
    """
    Parses GEO metadata to create binary labels for fertile (0) or infertile (1).
    """
    label_dict = {}
    print("Parsing metadata to create labels...")
    for gsm_name, gsm in gse.gsms.items():
        # Default to infertile (1) unless we find a specific keyword
        label = 1 
        characteristics = gsm.metadata.get('characteristics_ch1', [])
        
        # Iterate through each characteristic line to find the correct keyword
        for char_line in characteristics:
            # CORRECTED: The keyword for fertile samples in this dataset is 'condition: control'
            if "condition: control" in char_line.lower():
                label = 0 # Label as fertile (0)
                break
        label_dict[gsm_name] = label
    
    labels = pd.Series(label_dict).reindex(sample_order).dropna()
    print(f"Created labels for {len(labels)} samples.")
    print(f"Fertile (0) count: {(labels == 0).sum()}, Infertile (1) count: {(labels == 1).sum()}")
    return labels


def train_and_save_model():
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("--- Starting Model Training on REAL Data ---")
    raw_data, gse = fetch_data(TRAINING_ACCESSION_ID)
    if raw_data is None: return

    X, features, scaler = preprocess_training_data(raw_data)
    if X is None: return

    y = create_labels_from_metadata(gse, sample_order=X.index)
    X = X.reindex(y.index)
    if y.empty:
        print("No labels could be created. Aborting training."); return

    # Check for imbalance after labeling
    if y.nunique() < 2:
        print("Error: The dataset has only one class after labeling. Cannot train a meaningful model.")
        print("Please check the 'create_labels_from_metadata' function for this accession ID.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y) # Train on all available data for the final production model
    print("Model training complete.")

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy on a held-out test set: {accuracy:.2f}")

    print("Bundling model, feature list, and scaler into a single file...")
    model_artifacts = {
        'model': model,
        'features': features,
        'scaler': scaler
    }

    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print(f"--- Artifacts saved successfully to {MODEL_PATH} ---")

if __name__ == '__main__':
    train_and_save_model()

