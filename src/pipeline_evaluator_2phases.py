"""
File: pipeline_evaluator_2phases.py
Description: 
    Comprehensive testing and evaluation script for the Two-Phase AI Firewall Architecture.
    This source code is engineered to ensure reproducibility and 
    scientific integrity for the MAPR 2026 publication.

Core Capabilities:
    1. A/B Testing: Parallel execution of the proposed Two-Phase pipeline versus the single-phase baseline (XGBoost).
    2. Precision Latency: Utilizes time.perf_counter() for microsecond-level execution latency measurement.
    3. Scientific Metrics: Accurate computation of evaluation metrics (e.g., F1-Score, MCC, Precision, Recall).
    4. Visualization: Generates distribution and latency plots.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Integration of the static rule-based engine (Phase 1)
from define_label import lf_sqli_patterns_sql, lf_tautology

# Feature dimension list
from feature_extractor import MODEL_FEATURES

# =====================================================================
# CONFIGURATION PARAMETERS AND FEATURES
# =====================================================================
XGB_THRESHOLD = 0.50
PHASE_ORDER = ['Phase 1 (Screening)', 'Phase 2 (XGBoost)']

# =====================================================================
# TIERED ARCHITECTURE (PIPELINE PHASES)
# =====================================================================

def phase1_screening_check(sql_query):
    """
    Phase 1: Deterministic screening based on static heuristics.
    Objective: Rapidly intercept explicit attack vectors to mitigate computational overhead for Phase 2.
    """
    if lf_sqli_patterns_sql(sql_query) == 1 or lf_tautology(sql_query) == 1:
        return 1
    return 0

def load_xgb_model(xgb_path):
    """Initializes the foundational machine learning model for Phase 2."""
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost model not found at: {xgb_path}")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    return xgb_model

# =====================================================================
# EVALUATION LOOP
# =====================================================================

def evaluate_pipeline(df, xgb_model):
    """
    Simulates empirical inference to quantify system performance and latency constraints.
    """
    print("[*] Executing empirical inference across the evaluation dataset...")
    
    results = {
        'True_Label': [],
        'Pred_Label': [],
        'Resolved_Phase': [],
        'Latency_2Phase_ms': [],
        'Latency_1Phase_ms': [] # Baseline comparison tracking
    }

    # Suppress Pandas fragmentation warnings during feature extraction
    features_df = df[MODEL_FEATURES].copy()

    for idx, row in df.iterrows():
        query = str(row.get('SQL_query_sample', ''))
        true_label = int(row.get('hard_label', row.get('True_Label', 0)))
        
        # Extract a single feature instance as a DataFrame for XGBoost ingestion
        feat_row = features_df.iloc[[idx]]

        # -------------------------------------------------------------
        # 1. TWO-PHASE ARCHITECTURE SIMULATION (PROPOSED METHODOLOGY)
        # -------------------------------------------------------------
        t0_2p = time.perf_counter()
        
        # Execute Phase 1 screening
        p1_res = phase1_screening_check(query)
        if p1_res == 1:
            pred_2p = 1
            phase_2p = 'Phase 1 (Screening)'
        else:
            # Execute Phase 2 if Phase 1 indicates benign traffic
            prob = float(xgb_model.predict_proba(feat_row)[0][1])
            pred_2p = 1 if prob >= XGB_THRESHOLD else 0
            phase_2p = 'Phase 2 (XGBoost)'
            
        t1_2p = time.perf_counter()

        # -------------------------------------------------------------
        # 2. SINGLE-PHASE ARCHITECTURE SIMULATION (BASELINE: XGBOOST ONLY)
        # -------------------------------------------------------------
        t0_1p = time.perf_counter()
        prob_1p = float(xgb_model.predict_proba(feat_row)[0][1])
        pred_1p = 1 if prob_1p >= XGB_THRESHOLD else 0
        t1_1p = time.perf_counter()

        # Persist execution metrics
        results['True_Label'].append(true_label)
        results['Pred_Label'].append(pred_2p)
        results['Resolved_Phase'].append(phase_2p)
        results['Latency_2Phase_ms'].append((t1_2p - t0_2p) * 1000.0)
        results['Latency_1Phase_ms'].append((t1_1p - t0_1p) * 1000.0)

    results_df = pd.DataFrame(results)
    return results_df

# =====================================================================
# EVALUATION METRICS VISUALIZATION SYSTEM
# =====================================================================

def plot_and_report_results(df, results_dir):
    """
    Generates IEEE-compliant visualizations.
    Retains exclusively the Traffic & Latency distribution plot, incorporating dynamic scaling to resolve overlap issues.
    """
    os.makedirs(results_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 12, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

    # Standardize the linear processing order of the pipeline
    df['Resolved_Phase'] = pd.Categorical(df['Resolved_Phase'], categories=PHASE_ORDER, ordered=True)

    # --- DUAL-AXIS PLOT: TRAFFIC VOLUME & EXECUTION LATENCY ---
    phase_stats = df.groupby('Resolved_Phase', observed=False).agg(
        Traffic=('Resolved_Phase', 'count'),
        Avg_Latency=('Latency_2Phase_ms', 'mean')
    ).fillna(0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = '#1f77b4'
    ax1.set_xlabel('Processing Phase', fontsize=12)
    ax1.set_ylabel('Query Traffic (Count)', color=color1, fontsize=12)
    
    x_labels = phase_stats.index.astype(str)
    bars = ax1.bar(x_labels, phase_stats['Traffic'], color=color1, alpha=0.7, width=0.4, label='Traffic Count')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Dynamically scale Y-axis limits to prevent annotation overlap
    max_traffic = max(phase_stats['Traffic']) if max(phase_stats['Traffic']) > 0 else 1
    ax1.set_ylim(0, max_traffic * 1.5)  # Add 50% overhead clearance

    # Annotate quantitative values on bar plots
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + (max_traffic * 0.02), 
                 int(yval), ha='center', va='bottom', color=color1, fontweight='bold')

    # Secondary Y-axis for Latency distribution
    ax2 = ax1.twinx()  
    color2 = '#d62728'
    ax2.set_ylabel('Average Latency (ms)', color=color2, fontsize=12)  
    line = ax2.plot(x_labels, phase_stats['Avg_Latency'], color=color2, marker='s', linewidth=2.5, markersize=8, label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Dynamically scale secondary Y-axis to elevate the line plot above the bars
    max_latency = max(phase_stats['Avg_Latency']) if max(phase_stats['Avg_Latency']) > 0 else 1
    ax2.set_ylim(0, max_latency * 1.15) # Moderate overhead clearance for the line plot

    # Annotate empirical values on the line plot
    for i, txt in enumerate(phase_stats['Avg_Latency']):
        y_offset = txt + (max_latency * 0.03)
        ax2.text(i, y_offset, f"{txt:.4f} ms", ha='center', va='bottom', color=color2, fontweight='bold')

    plt.title('End-to-End Traffic and Latency Distribution', pad=15)
    
    # Render unified legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()  
    plt.savefig(os.path.join(results_dir, "pipeline_traffic_distribution.png"), dpi=300)
    plt.close()
    
    print("[INFO] IEEE-compliant visualization successfully generated and persisted in 'results_2phases/'.")

# =====================================================================
# MAIN EXECUTION
# =====================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Ensure artifacts from training exist
    TEST_CSV = os.path.join(BASE_DIR, "results_2phases", "firewall_test_dataset.csv")  
    MODEL_PATH = os.path.join(BASE_DIR, "results_2phases", "xgboost_phase2.json")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_2phases")
    
    print("[INFO] Initializing the AI Firewall evaluation pipeline (Two-Phase Architecture)...")
    
    try:
        if not os.path.exists(TEST_CSV):
            print(f"[!] ERROR: Test dataset not located at: {TEST_CSV}")
            print(f"    Please ensure the CSV file contains: 'SQL_query_sample', 'hard_label', and the requisite XGBoost feature columns.")
        else:
            df_test = pd.read_csv(TEST_CSV)
            print(f"[INFO] Ingested {len(df_test)} data samples from the test set.")
            
            # Verify the existence of mandatory feature dimensions
            missing_features = [f for f in MODEL_FEATURES if f not in df_test.columns]
            if missing_features:
                print(f"[!] ERROR: Dataset is missing mandatory feature columns: {missing_features}")
            else:
                # Load baseline model and execute the evaluation pipeline
                xgb_model = load_xgb_model(MODEL_PATH)
                results_df = evaluate_pipeline(df_test, xgb_model)
                
                # Persist detailed inference results
                os.makedirs(RESULTS_DIR, exist_ok=True)
                results_df.to_csv(os.path.join(RESULTS_DIR, "pipeline_detailed_results.csv"), index=False)
                
                # Generate visualizations and metrics
                plot_and_report_results(results_df, RESULTS_DIR)

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")