import os
from flask import Flask, render_template, request, jsonify
import logging

# --- Correctly configure paths for templates and static files ---
# This ensures Flask can find your HTML, CSS, and JS files.
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

# --- App Initialization ---
# Note: Use relative paths here after defining the absolute paths above.
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['STATIC_FOLDER'] = static_dir
app.config['TEMPLATES_AUTO_RELOAD'] = True

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)

# --- Dynamic Import of Core Modules ---
# This is a robust way to handle imports in a packaged Flask application.
try:
    from .core import data_fetcher, preprocessor, predictor
except (ImportError, SystemError):
    from core import data_fetcher, preprocessor, predictor


@app.route('/')
def index():
    """ Renders the main page. """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend.
    Fetches data, preprocesses it, and returns the model's prediction.
    """
    data = request.get_json()
    accession_id = data.get('accession_id')

    if not accession_id:
        return jsonify({'error': 'Accession ID is required.'}), 400

    try:
        # --- Step 1: Fetch Data ---
        logging.info(f"Fetching data for {accession_id}...")
        raw_data, _ = data_fetcher.fetch_data(accession_id)
        if raw_data is None or raw_data.empty:
            return jsonify({'error': f'Failed to fetch or parse data for {accession_id}. The dataset might be private, unavailable, or in an unsupported format.'}), 500

        # --- Step 2: Preprocess Data ---
        logging.info("Preprocessing data for prediction...")
        processed_data = preprocessor.preprocess_prediction_data(raw_data)
        if processed_data is None:
             return jsonify({'error': 'The provided dataset does not contain enough overlapping features (genes) with the training data to make a reliable prediction.'}), 400

        # --- Step 3: Get Prediction ---
        logging.info("Making prediction...")
        prediction_result = predictor.predict(processed_data)

        return jsonify(prediction_result)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        # Return a user-friendly but informative error
        return jsonify({'error': f"An unexpected error occurred. Please check the logs. Details: {str(e)}"}), 500


if __name__ == '__main__':
    # Runs the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)

