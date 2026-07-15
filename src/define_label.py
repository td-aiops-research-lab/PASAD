"""
File: define_label.py
Description: A comprehensive suite of Regular Expression (Regex) patterns and heuristic rules, 
             synergized with empirical execution metrics, designed to assign labels 
             for the training dataset.
"""

import pandas as pd
import numpy as np
import re

# =====================================================================
# 1. ANOMALY DETECTION VIA EMPIRICAL EXECUTION METRICS (GROUND TRUTH)
# =====================================================================
def lf_execution_metrics(row):
    """
    Assigns performance-based labels utilizing empirical statistics extracted from the MySQL Performance Schema.
    Scientific heuristics:
    1. Anomalous (1): High latency coupled with extensive non-indexed table scans or disk-based temporary table instantiations.
    2. Erroneous (1): Presence of syntactic or execution errors (SUM_ERRORS > 0).
    3. Benign (0): Rapid execution utilizing proper index lookups.
    4. Abstain (-1): Borderline metrics lacking definitive classification.
    """
    try:
        latency = float(row.get('avg_latency_ms', 0))
        rows_examined = float(row.get('avg_rows_examined', 0))
        rows_sent = float(row.get('avg_rows_sent', 1)) # Mitigate division by zero exception
        disk_tmp = float(row.get('tmp_disk_tables', 0))
        no_index = float(row.get('SUM_NO_INDEX_USED', 0))
        errors = float(row.get('SUM_ERRORS', 0))

        # Detect execution errors
        if errors > 0:
            return 1
            
        ratio_examined_sent = rows_examined / (rows_sent if rows_sent > 0 else 1)
        
        # Definitive performance anomaly
        if (latency > 500.0) or (disk_tmp > 0) or (ratio_examined_sent > 10000 and no_index > 0):
            return 1 
            
        # Optimal execution with index utilization -> Benign
        if latency < 50.0 and no_index == 0:
            return 0 
            
    except Exception:
        pass
        
    return -1 # Indeterminate state (Abstain)

# =====================================================================
# 2. SQL INJECTION (SQLi) PATTERN RECOGNITION
# =====================================================================
def lf_sqli_patterns(row):
    sql = str(row.get("SQL_query_sample", row.get("DIGEST_TEXT", ""))).upper()
    return lf_sqli_patterns_sql(sql)

def lf_sqli_patterns_sql(sql):
    if not isinstance(sql, str):
        return -1
    sqli_patterns = [
        r"(?:')\s*(?:OR|AND)\s*(?:\d+=\d+|'[^']*'='[^']*'|\"[^\"]*\"=\"[^\"]*\")",
        r"(?:--|#|\/\*).*$",
        r"\b(?:UNION(?: ALL)?\s+SELECT)\b",
        r"\b(?:EXEC|EXECUTE)\b\s*\w+",
        r"\b(?:WAITFOR\s+DELAY|SLEEP|BENCHMARK)\b",
        r"';\s*(?:DROP|UPDATE|DELETE|INSERT|TRUNCATE)\b"
    ]
    for pattern in sqli_patterns:
        if re.search(pattern, sql):
            return 1
    return -1

# =====================================================================
# 3. TAUTOLOGY DETECTION (ALWAYS-TRUE CONDITIONS)
# =====================================================================
def lf_tautology(row_or_sql):
    """
    Upgraded for flexible data ingestion:
    - Accepts a Pandas Row (utilized during the dataset construction pipeline).
    - Accepts a raw SQL string (utilized during real-time Pipeline Evaluator execution).
    """
    # Handle raw string input (from real-time Pipeline Evaluator)
    if isinstance(row_or_sql, str):
        sql = row_or_sql.upper()
    # Handle Pandas Row / Dictionary input (from dataset_setup.py)
    else:
        sql = str(row_or_sql.get("DIGEST_TEXT", row_or_sql.get("SQL_query_sample", ""))).upper()

    if re.search(r"\bWHERE\b", sql):
        tautology_patterns = [
            r"\b(\d+)\s*=\s*\1\b",
            r"'([^']*)'\s*=\s*'\1'",
            r'\b"([^"]*)"\s*=\s*"\1"',
            r"\b1\s*=\s*1\b"
        ]
        for pattern in tautology_patterns:
            if re.search(pattern, sql):
                return 1
    return -1

# =====================================================================
# 4. DANGEROUS DML / DDL DETECTION AND WHITELIST HEURISTICS
# =====================================================================
def lf_dangerous_ddl(row):
    sql = str(row.get("DIGEST_TEXT", "")).upper()
    if re.search(r"^\s*(DROP|TRUNCATE|ALTER|GRANT|REVOKE)\b", sql):
        return 1  
    return -1

def lf_unsafe_dml(row):
    sql = str(row.get("DIGEST_TEXT", "")).upper()
    if re.search(r"^\s*(UPDATE|DELETE)\b", sql) and not re.search(r"\bWHERE\b", sql):
        return 1
    return -1

def lf_safe_crud(row):
    sql = str(row.get("DIGEST_TEXT", "")).upper()
    if re.search(r"^\s*SELECT\b.*?\bFROM\b", sql):
        if lf_sqli_patterns_sql(sql) == -1 and lf_tautology(row) == -1:
            return 0 
    return -1