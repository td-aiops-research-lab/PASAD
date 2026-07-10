"""
File: data_anonymizer_aiops.py
Description: Orchestrates the automated, synchronous anonymization pipeline for the AIOps firewall training dataset.
             It leverages the core anonymizer module to process and sanitize massive SQL query samples dynamically,
             generating an academic cryptographic report upon successful dataset transformation to ensure training safety and privacy.
"""

import pandas as pd
from tqdm import tqdm
import os

# Import reusable core module
from core_sql_anonymizer import CoreSQLAnonymizer


def process_dataset_anonymization(input_csv, output_csv):
    """
    Processes the anonymization of the entire dataset utilized for model training.
    """
    if not os.path.exists(input_csv):
        print(
            f"[!] ERROR: Input file {input_csv} not found. Please verify the file path."
        )
        return

    print(f"[*] Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Initialize the core anonymizer object
    anonymizer = CoreSQLAnonymizer(mask_literals=True)

    # Target columns containing SQL queries requiring anonymization
    target_columns = ["SCHEMA_NAME", "DIGEST_TEXT", "SQL_query_sample"]

    print("[*] Initiating comprehensive synchronous anonymization...")

    # Enable tqdm's progress_apply for visualization of the execution progress
    tqdm.pandas()

    for col in target_columns:
        if col in df.columns:
            print(f"[*] Processing column: {col}")
            # Apply the core module's anonymize_sql function to the dataset
            df[col] = df[col].progress_apply(anonymizer.anonymize_sql)

    # Persist the processed dataset to a file
    df.to_csv(output_csv, index=False)
    print(f"\n[*] COMPLETED! Anonymized dataset persisted at: {output_csv}")

    # =====================================================================
    # ACADEMIC REPORT FOR PUBLICATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANONYMIZATION REPORT (SCIENTIFIC METHODOLOGY):")
    print("1. Methodology: Deterministic Pseudonymization Protocol.")
    print(
        f"2. Structural Preservation: Retained {len(anonymizer.sql_keywords)} core SQL Keywords."
    )
    print(
        f"3. Identifier Pseudonymization (Tables/Columns): Generated {len(anonymizer._hash_cache)} SHA-256 truncated hashes."
    )
    print("4. Literal Processing:")
    print("   - Strings transformed into <STR> tags.")
    print("   - Numbers transformed into <NUM> tags.")
    print("=> Conclusion: Mitigated data leakage for off-site telemetry training.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Read from the labeled dataset generated in the previous step
    INPUT_FILE = "data/firewall_training_dataset_raw.csv"
    OUTPUT_FILE = "data/firewall_training_dataset_anonymized.csv"

    process_dataset_anonymization(INPUT_FILE, OUTPUT_FILE)