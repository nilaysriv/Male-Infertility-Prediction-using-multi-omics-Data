# Male-Infertility-Prediction-using-multi-omics-Data
This project is a Python-based tool designed to predict male infertility using machine learning and public genomics data. The script provides an end-to-end pipeline, from data retrieval and preprocessing to model training, evaluation, and interpretation.

The current implementation uses gene expression (transcriptomics) data from the NCBI Gene Expression Omnibus (GEO) to build a predictive model.

# Features

Automated Data Retrieval: Downloads and parses real-world medical datasets directly from the NCBI GEO repository.

End-to-End Pipeline: Implements a complete workflow, including preprocessing, feature selection, and model training.

Advanced Modeling: Uses a Random Forest classifier with hyperparameter tuning to find the best-performing model.

Model Evaluation: Generates a full evaluation of the model, including an accuracy score, AUC score, classification report, and a confusion matrix.

Biomarker Discovery: Uses SHAP (SHapley Additive exPlanations) to interpret the model's decisions and identify the key genes that are most predictive of infertility.

# How It Works

The script is currently configured to use dataset GSE45887 from GEO. This dataset contains gene expression data from sperm samples of 20 fertile and 20 infertile (asthenozoospermic) men, providing a solid foundation for a binary classification task.

The model learns to identify complex patterns in the expression levels of thousands of genes to distinguish between the two groups.

# Getting Started

Prerequisites

Python 3.7 or higher

Installation & Running

The script is designed to be self-contained and will automatically install all necessary Python dependencies when you run it for the first time.

Clone the repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name


Run the script:

python infertility_predictor_geo.py


The script will first run the install_dependencies() function, which will pip install numpy, pandas, scikit-learn, matplotlib, seaborn, shap, and geoparse. After dependencies are installed, the main pipeline will execute.

# The Pipeline

The script executes the following steps in order:

Dependencies: Checks for and installs all required libraries.

Data Retrieval (retrieve_geo_data):

Downloads the GSE45887 dataset from NCBI GEO.

Parses the metadata to create the target labels (0 = Fertile, 1 = Infertile).

Parses the expression data into a feature matrix (samples x genes).

Maps probe IDs to gene symbols where possible.

Preprocessing (preprocess_data):

Imputes any missing values using the median.

Scales the data using StandardScaler so that all features have a mean of 0 and a standard deviation of 1. This is crucial for model performance.

Feature Selection (select_features):

Uses Recursive Feature Elimination (RFE) with a Random Forest to select the top 50 most predictive features (genes) from the thousands available. This reduces noise and computation time.

Train-Test Split:

Splits the data into a training set (to teach the model) and a test set (to evaluate its performance on unseen data).

Model Training (train_model):

Uses GridSearchCV to find the best hyperparameters for a RandomForestClassifier, optimizing for the AUC score.

Model Evaluation (evaluate_model):

Tests the best model on the held-out test set.

Prints the accuracy, AUC, and classification report.

Generates and displays a Confusion Matrix and an ROC Curve.

Model Interpretation (interpret_model):

Uses shap.TreeExplainer to understand why the model makes its predictions.

Generates a SHAP summary plot to show the top predictive genes and their impact on the prediction.

# Future Work: Towards Multi-Omics

This project currently serves as a robust template for transcriptomics. The original goal is to build a multi-omics tool. This pipeline can be extended to integrate data from other biological layers:

Genomics (WES/WGS): Data on SNPs and mutations.

Epigenomics (EPIC arrays): Data on DNA methylation patterns.

Proteomics (Mass Spec): Data on protein abundance.

Future development will focus on integrating these diverse data types into a single, more powerful predictive model.
