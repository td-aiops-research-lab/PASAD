"""
File: synthetic_generator.py
Description: Data synthesis engine for generating Performance Anomalies within the TPC-H schema, 
             incorporating empirical validation via real-time EXPLAIN execution plans.
"""

import random
import pandas as pd
from feature_extractor import get_db_connection

def generate_explainable_performance_anomalies(target_size, db_conn):
    """
    Synthesizes computationally intensive TPC-H queries. 
    MANDATORY: Ensures successful empirical validation via the EXPLAIN command to guarantee structural integrity.
    """
    print(f"[*] Synthesizing {target_size} Performance Anomaly samples within the 'tpch' schema...")
    
    valid_queries = []
    cursor = db_conn.cursor()
    
    try:
        cursor.execute("USE tpch;")
    except Exception as e:
        print(f"[!] Failed to select the target database 'tpch'. Error details: {e}")
        return pd.DataFrame()
    
    tables = ['customer', 'orders', 'lineitem', 'part', 'supplier', 'partsupp', 'nation', 'region']
    string_cols = ['c_name', 'c_address', 'c_phone', 'o_comment', 'o_clerk', 'l_comment', 'p_name', 's_name']
    
    while len(valid_queries) < target_size:
        limit_val = random.randint(10000, 50000)
        c1 = random.choice(string_cols)
        
        templates = [
            "SELECT * FROM customer, orders WHERE customer.c_mktsegment = 'BUILDING'",
            "SELECT l_orderkey, l_extendedprice FROM lineitem ORDER BY l_comment DESC LIMIT 100",
            f"SELECT * FROM part WHERE UPPER({c1}) LIKE '%STEEL%'",
            "SELECT c_name FROM customer WHERE c_acctbal > (SELECT AVG(o_totalprice) FROM orders WHERE orders.o_custkey = customer.c_custkey)",
            "SELECT * FROM supplier WHERE s_comment LIKE '%special%requests%'"
        ]
        
        sql = random.choice(templates)
        
        try:
            # Mandate immediate EXPLAIN execution to empirically verify syntactic and semantic correctness
            cursor.execute(f"EXPLAIN FORMAT=JSON {sql}")
            cursor.fetchone()
            
            valid_queries.append({
                "SCHEMA_NAME": "tpch",
                "DIGEST_TEXT": sql,
                "SQL_query_sample": sql,
                "source": "TPCH_EXPLAINED_PERF_ANOMALY"
            })
        except Exception:
            continue # Discard the synthesized query if it fails structural validation against the TPC-H schema

    cursor.close()
    return pd.DataFrame(valid_queries)