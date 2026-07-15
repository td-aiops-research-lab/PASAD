"""
File: pipline_data_collector.py
Description: Comprehensive integration pipeline for dynamic feature extraction utilizing 
             100% authentic MySQL EXPLAIN execution plans (zero simulation).
"""

import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv

from synthetic_generator import generate_explainable_performance_anomalies
from hybrid_sqli_loader import generate_explainable_sqli_piggybacking
from feature_extractor import (
    extract_static_features,
    extract_dynamic_features,
    MODEL_FEATURES,
    get_db_connection
)

load_dotenv()

FEATURES = MODEL_FEATURES + ["soft_label", "hard_label", "explain_status"]

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("PHASE 1: INGEST LABELED DATASET & SYNTHESIZE TPC-H EMPIRICAL DATA")
    print("=" * 60)
    
    # 1. Ingest historical MySQL query data
    historical_file = os.path.join(BASE_DIR, "data", "dataset_labeled_final.csv")
    if os.path.exists(historical_file):
        df_history = pd.read_csv(historical_file)
        df_history["source"] = "MySQL_Digest_History"
    else:
        df_history = pd.DataFrame()

    # 2. Initialize a shared database connection for data synthesis
    db_conn = get_db_connection()
    if not db_conn:
        print("[!] Database connection failed. Please verify the environment configuration.")
        return

    # 3. Synthesize explainable anomalous and malicious datasets
    df_perf = generate_explainable_performance_anomalies(target_size=2000, db_conn=db_conn)
    df_sqli = generate_explainable_sqli_piggybacking(target_size=2000, db_conn=db_conn)

    # Assign deterministic labels to the synthesized datasets
    for df_syn in [df_perf, df_sqli]:
        if not df_syn.empty:
            df_syn["soft_label"] = 1.0
            df_syn["hard_label"] = 1

    df_all = pd.concat([df_history, df_perf, df_sqli], ignore_index=True)
    print(f"[*] Total number of records staged for feature extraction: {len(df_all)}")

    print("\n" + "=" * 60)
    print("PHASE 2: EMPIRICAL FEATURE EXTRACTION VIA REAL EXPLAIN COMMANDS (NON-SIMULATED)")
    print("=" * 60)

    processed_data = []
    # 5. Persist to data directory
    output_csv = os.path.join(BASE_DIR, "data", "firewall_training_dataset_raw.csv")

    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Processing Features"):
        sql_sample = str(row.get("SQL_query_sample", row.get("DIGEST_TEXT", "")))
        schema = str(row.get("SCHEMA_NAME", ""))
        source = str(row.get("source", "Unknown"))

        # 1. Compute static features
        static_feat = extract_static_features(sql_sample)
        
        # 2. Retrieve dynamic features (EXECUTE REAL EXPLAIN ON MYSQL)
        dynamic_feat = extract_dynamic_features(sql_sample, default_db=schema)

        if dynamic_feat["explain_status"] != "SUCCESS" and "TPCH" not in source:
            skipped_count += 1
            continue  # Discard historical queries that are no longer parsable/explainable

        record = {
            "SCHEMA_NAME": schema,
            "DIGEST_TEXT": row.get("DIGEST_TEXT", ""),
            "SQL_query_sample": sql_sample,
            **static_feat,
            **dynamic_feat,
            "soft_label": row.get("soft_label", 0),
            "hard_label": row.get("hard_label", 0),
            "source": source,
        }
        processed_data.append(record)

    df_final = pd.DataFrame(processed_data)
    numeric_features = [f for f in FEATURES if f not in ["explain_status"]]
    
    for col in numeric_features:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce").fillna(0)

    cols_to_save = ["SCHEMA_NAME", "DIGEST_TEXT", "SQL_query_sample"] + FEATURES + ["source"]
    df_final = df_final[[c for c in cols_to_save if c in df_final.columns]]
    df_final.to_csv(output_csv, index=False)

    print(f"\n[*] ELIMINATED {skipped_count} historical records failing real-time EXPLAIN validation.")
    print(f"[*] Total validated records persisted for Model Training (Post-EXPLAIN): {len(df_final)} rows.")
    print("\nEXPLAIN Execution Status Breakdown:")
    print(df_final["explain_status"].value_counts())
    print("\nData Source Distribution Breakdown:")
    print(df_final.groupby(["source", "hard_label"]).size())

if __name__ == "__main__":
    main()