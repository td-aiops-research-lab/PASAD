"""
File: training_experiment_core_static.py
Description: 
    Training and evaluation script for the classification model (Phase 2 - AI Firewall).
    Engineered strictly adhering to scientific MLOps standards, ensuring 100% 
    logical synchronization with pipeline_evaluator.py.

Core Execution Pipeline:
    1. Standardized feature extraction (Explicitly defined MODEL_FEATURES).
    2. Rigorous data partitioning: Train (70%), Validation (15%), Test (15%).
    3. Pipeline instantiation with StandardScaler for Baseline models (including DNN) and XGBoost.
    4. Hyperparameter optimization for XGBoost utilizing 5-Fold Cross-Validation.
    5. Objective evaluation on the hold-out Test set applying a fixed Threshold (0.50) 
       to guarantee consistency with the End-to-End architecture.
    6. Micro-latency inference measurement leveraging pure NumPy arrays.
    7. Artifact exportation: Metrics CSV, Joblib PR-curves, XGBoost model (JSON), and Scaler.
"""

import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, auc, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# # Import the standardized feature set from the system configuration
# from feature_extractor import MODEL_FEATURES

MODEL_FEATURES = [
    "query_length", "num_joins", "num_subqueries", "has_wildcard", "has_union",
    "has_groupby_orderby", "has_comment"
]

warnings.filterwarnings("ignore")

# =====================================================================
# 1. EXPERIMENTAL CONFIGURATION
# =====================================================================
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results_2phases_static")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed EVALUATION_THRESHOLD to ensure absolute synchronization with pipeline_evaluator.py
EVALUATION_THRESHOLD = 0.50

# =====================================================================
# 2. DATA PREPARATION AND PARTITIONING
# =====================================================================
def load_and_prepare_data(file_path: str):
    print("[INFO] Ingesting and partitioning the dataset...")
    df = pd.read_csv(file_path)
    
    # Filter out database execution errors and entries with missing values
    df = df[~df['explain_status'].astype(str).str.startswith('DB_ERROR')]
    df = df.dropna(subset=MODEL_FEATURES + ['hard_label', 'SQL_query_sample'])
    
    # DATA PARTITIONING: Isolate the independent Test set (15%) from the remainder (85%)
    df_temp, df_test = train_test_split(df, test_size=0.15, stratify=df['hard_label'], random_state=42)
    
    # DATA PARTITIONING: Split the remainder into Training (70%) and Validation (15%) sets
    df_train, df_val = train_test_split(df_temp, test_size=0.1764, stratify=df_temp['hard_label'], random_state=42)
    
    # Persist the isolated Test set to a distinct CSV for Pipeline Evaluator consumption
    test_csv_path = os.path.join(RESULTS_DIR, "firewall_test_dataset.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f" -> Independent Test Set saved to: {test_csv_path}")
    
    print(f" -> Train set:      {df_train.shape[0]} samples")
    print(f" -> Validation set: {df_val.shape[0]} samples")
    print(f" -> Test set:       {df_test.shape[0]} samples")
    
    X_train, y_train = df_train[MODEL_FEATURES], df_train['hard_label']
    X_val, y_val = df_val[MODEL_FEATURES], df_val['hard_label']
    X_test, y_test = df_test[MODEL_FEATURES], df_test['hard_label']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# =====================================================================
# 3. MODEL DEFINITION AND TRAINING
# =====================================================================
def define_models():
    """Defines Baseline models (including Multilayer Perceptron) and configures the hyperparameter grid for XGBoost."""
    models = {
        "Dummy": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Multilayer Perceptron (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42, 
            early_stopping=True
        )
    }
    
    xgb_param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0]
    }
    return models, xgb_param_grid

def evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n[INFO] Starting model training and evaluation...")
    models, xgb_param_grid = define_models()
    results = {}
    pr_curves = {}

    # Convert X_test to a pure NumPy array to measure pure inference latency (bypassing Pandas structural overhead)
    X_test_np = X_test.to_numpy()

    # 3.1. BASELINE MODEL TRAINING
    for name, model in models.items():
        print(f"[*] Training {name}...")
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        pipeline.fit(X_train, y_train)
        
        # Evaluate on the Test set (Applying the unified EVALUATION_THRESHOLD)
        t0 = time.perf_counter()
        y_test_proba = pipeline.predict_proba(X_test_np)[:, 1]
        t1 = time.perf_counter()
        
        latency_us = ((t1 - t0) / len(X_test_np)) * 1_000_000
        y_test_pred = (y_test_proba >= EVALUATION_THRESHOLD).astype(int)
        
        test_precisions, test_recalls, _ = precision_recall_curve(y_test, y_test_proba)
        results[name] = {
            "F1-Score": f1_score(y_test, y_test_pred),
            "MCC": matthews_corrcoef(y_test, y_test_pred),
            "PR-AUC": auc(test_recalls, test_precisions),
            "Latency (us)": latency_us
        }
        pr_curves[name] = {"precision": test_precisions, "recall": test_recalls, "auc": auc(test_recalls, test_precisions)}
        print(f" -> Test F1-Score: {results[name]['F1-Score']:.4f} | Latency: {latency_us:.2f} µs")

    # 3.2. XGBOOST TRAINING AND HYPERPARAMETER OPTIMIZATION (PROPOSED MODEL)
    print("\n[*] Training XGBoost (Proposed Model) with Hyperparameter Tuning...")
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_pipeline, xgb_param_grid, n_iter=20, 
        scoring=make_scorer(f1_score), cv=cv, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_xgb_pipeline = search.best_estimator_
    
    # Evaluate on the Test set utilizing NumPy arrays
    t0 = time.perf_counter()
    y_test_proba = best_xgb_pipeline.predict_proba(X_test_np)[:, 1]
    t1 = time.perf_counter()
    
    latency_us = ((t1 - t0) / len(X_test_np)) * 1_000_000
    y_test_pred = (y_test_proba >= EVALUATION_THRESHOLD).astype(int)
    
    test_precisions, test_recalls, _ = precision_recall_curve(y_test, y_test_proba)
    name = "XGBoost (Proposed)"
    results[name] = {
        "F1-Score": f1_score(y_test, y_test_pred),
        "MCC": matthews_corrcoef(y_test, y_test_pred),
        "PR-AUC": auc(test_recalls, test_precisions),
        "Latency (us)": latency_us
    }
    pr_curves[name] = {"precision": test_precisions, "recall": test_recalls, "auc": auc(test_recalls, test_precisions)}
    print(f" -> Test F1-Score: {results[name]['F1-Score']:.4f} | Latency: {latency_us:.2f} µs")

    return results, pr_curves, best_xgb_pipeline

# =====================================================================
# 4. ARTIFACT EXPORTATION AND VISUALIZATION (CHARTS)
# =====================================================================
def plot_pr_curves(pr_curves):
    """Generates Precision-Recall (PR) curves for all evaluated models."""
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    
    # Assign distinct color palettes, including the newly added MLP model
    colors = ['gray', 'blue', 'green', 'purple', 'red']
    for (name, data), color in zip(pr_curves.items(), colors):
        plt.plot(data['recall'], data['precision'], label=f'{name} (AUC = {data["auc"]:.4f})', color=color, linewidth=2.5)
        
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (Independent Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower left', fontsize=11)
    
    chart_path = os.path.join(RESULTS_DIR, "precision_recall_curves.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> PR Curves chart saved to: {chart_path}")

def export_results(results, pr_curves, best_xgb_pipeline):
    print("\n[INFO] Exporting models and evaluation results...")
    
    # Persist the evaluation metrics to a CSV file
    df_results = pd.DataFrame.from_dict(results, orient="index")
    csv_path = os.path.join(RESULTS_DIR, "model_evaluation_metrics.csv")
    df_results.to_csv(csv_path, index_label="Model")
    print(f" -> Metrics (CSV) saved to: {csv_path}")

    # Serialize plotting data using Joblib
    joblib_path = os.path.join(RESULTS_DIR, "pr_curves_data.joblib")
    joblib.dump(pr_curves, joblib_path)
    
    # Generate and save the visualizations
    plot_pr_curves(pr_curves)

    # Extract and serialize the Scaler (Joblib) and XGBoost model (JSON)
    scaler = best_xgb_pipeline.named_steps['scaler']
    xgb_model = best_xgb_pipeline.named_steps['clf']
    
    scaler_path = os.path.join(RESULTS_DIR, "scaler_phase2.joblib")
    model_path = os.path.join(RESULTS_DIR, "xgboost_phase2.json")
    
    joblib.dump(scaler, scaler_path)
    xgb_model.save_model(model_path)
    print(f" -> Scaler object saved to:  {scaler_path}")
    print(f" -> XGBoost model saved to:  {model_path}")

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Load the cryptographically anonymized preprocessed dataset
    DATA_FILE = os.path.join(BASE_DIR, "data", "firewall_training_dataset_anonymized.csv")
    
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Dataset not found: {DATA_FILE}")
        exit(1)
        
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(DATA_FILE)
    results, pr_curves, best_xgb_pipeline = evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)
    export_results(results, pr_curves, best_xgb_pipeline)
    print("\n[SUCCESS] Phase 2 training complete. Model is ready for Holistic Evaluation.")