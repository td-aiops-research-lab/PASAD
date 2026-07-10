"""
File: dataset_setup.py
Description: Orchestrates data preprocessing and label aggregation.
             This module assigns labels based on a composite of structural rules and
             historical execution failures, assigning the highest priority to execution metrics.
"""

import pandas as pd
import numpy as np

def clean_mysql_exported_data(df):
    print(f"[*] Initial dataset volume (row count): {len(df)}")
    df.replace("NULL", np.nan, inplace=True)
    df = df.dropna(subset=["DIGEST_TEXT"]).copy()
    print(f"[*] Dataset volume post-cleansing (null SQL entries removed): {len(df)}")
    return df

from define_label import (
    lf_execution_metrics,
    lf_sqli_patterns,
    lf_tautology,
    lf_dangerous_ddl,
    lf_unsafe_dml,
    lf_safe_crud
)

def apply_labeling_functions(df):
    print("[*] Applying composite structural rules and execution metrics...")

    df["lf_metrics"] = df.apply(lf_execution_metrics, axis=1)
    df["lf_sqli"] = df.apply(lf_sqli_patterns, axis=1)
    df["lf_tautology"] = df.apply(lf_tautology, axis=1)
    df["lf_ddl"] = df.apply(lf_dangerous_ddl, axis=1)
    df["lf_unsafe_dml"] = df.apply(lf_unsafe_dml, axis=1)
    df["lf_safe"] = df.apply(lf_safe_crud, axis=1)

    def calculate_final_label(row):
        # Operational execution cost limits
        if row["lf_metrics"] == 1: return 1
        if row["lf_metrics"] == 0: return 0
        
        # Structural rules and explicit SQL injection signatures
        if row["lf_sqli"] == 1: return 1
        if row["lf_tautology"] == 1: return 1
        if row["lf_ddl"] == 1: return 1
        if row["lf_unsafe_dml"] == 1: return 1

        # Operational baseline queries are labeled as normal (y=0)
        if row["lf_safe"] == 0: return 0

        # Default normal label
        return 0  

    df["hard_label"] = df.apply(calculate_final_label, axis=1)

    print("\n--- FINAL LABEL AGGREGATION RESULTS ---")
    print(df["hard_label"].value_counts())

    return df

if __name__ == "__main__":
    INPUT_FILE = "digest_text_with_samples.csv"
    OUTPUT_FILE = "dataset_labeled_final.csv"

    try:
        df_raw = pd.read_csv(INPUT_FILE)
        df_cleaned = clean_mysql_exported_data(df_raw)
        df_labeled = apply_labeling_functions(df_cleaned)
        df_labeled.to_csv(OUTPUT_FILE, index=False)
        print(f"[*] Aggregation complete. Labeled dataset persisted to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"[!] Execution Error: {e}")