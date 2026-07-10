"""
File: hybrid_sqli_loader.py
Description: Generates explainable Piggybacked SQL Injection (SQLi) datasets utilizing the TPC-H schema.
"""

import random
import pandas as pd

def generate_explainable_sqli_piggybacking(target_size, db_conn):
    """
    Synthesizes SQL Injection queries (incorporating UNION-based, Tautology, or Blind SQLi techniques) 
    that are successfully parsable by the EXPLAIN statement.
    """
    print(f"[*] Synthesizing {target_size} Piggybacked SQLi samples within the 'tpch' schema...")
    
    valid_queries = []
    cursor = db_conn.cursor()
    
    try:
        cursor.execute("USE tpch;")
    except Exception as e:
        print(f"[!] Failed to select the target database 'tpch'. Error details: {e}")
        return pd.DataFrame()
    
    while len(valid_queries) < target_size:
        cust_id = random.randint(1, 1000)
        
        templates = [
            f"SELECT c_name, c_address FROM customer WHERE c_custkey = {cust_id} UNION ALL SELECT user, password FROM mysql.user",
            f"SELECT * FROM orders WHERE o_orderkey = {cust_id} AND SLEEP(1) = 0",
            f"SELECT * FROM part WHERE p_partkey = {cust_id} OR 1=1 -- ",
            f"SELECT * FROM supplier WHERE s_suppkey = {cust_id} AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT DATABASE())))",
            f"SELECT * FROM lineitem WHERE l_orderkey = {cust_id} UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
        ]
        
        sql = random.choice(templates)
        
        try:
            # Attempt query execution plan extraction via EXPLAIN to validate syntax and explainability
            cursor.execute(f"EXPLAIN FORMAT=JSON {sql}")
            cursor.fetchone()
            
            valid_queries.append({
                "SCHEMA_NAME": "tpch",
                "DIGEST_TEXT": sql,
                "SQL_query_sample": sql,
                "source": "TPCH_EXPLAINED_SQLI"
            })
        except Exception:
            continue

    cursor.close()
    return pd.DataFrame(valid_queries)