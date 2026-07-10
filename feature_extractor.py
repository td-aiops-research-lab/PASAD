"""
File: feature_extractor.py
Description: Orchestrates the extraction of both static and dynamic features from SQL queries. 
             Integrates an automated database discovery algorithm utilizing the information_schema 
             to dynamically resolve target schemas for execution plan analysis.
"""

import json
import pymysql
import os
import re
from dotenv import load_dotenv

load_dotenv()

MODEL_FEATURES = [
    "query_length", "num_joins", "num_subqueries", "has_wildcard", "has_union",
    "has_groupby_orderby", "has_comment", "query_cost", "estimated_rows",
    "is_full_table_scan", "key_used",
]

_db_conn = None
_current_db = None

def get_db_connection(db_config=None, dynamic_db=""):
    global _db_conn, _current_db
    if _db_conn is None:
        try:
            _db_conn = pymysql.connect(
                host=os.environ.get("DB_HOST", "localhost"),
                user=os.environ.get("DB_USER", "root"),
                password=os.environ.get("DB_PASSWORD", ""),
                database=os.environ.get("DB_NAME", ""),
                cursorclass=pymysql.cursors.DictCursor
            )
            _current_db = os.environ.get("DB_NAME", "")
            print("[*] MySQL database connection successfully established.")
        except Exception as e:
            print(f"[!] Database connection failure: {e}")
            return None
            
    if dynamic_db and dynamic_db != _current_db:
        try:
            with _db_conn.cursor() as cursor:
                cursor.execute(f"USE `{dynamic_db}`")
            _current_db = dynamic_db
        except:
            pass
            
    return _db_conn

def extract_db_from_sql(sql_text):
    pattern = r'(?i)\b(?:FROM|JOIN|UPDATE|INTO)\s+`?([a-zA-Z0-9_]+)`?\.'
    match = re.search(pattern, sql_text)
    if match:
        return match.group(1)
    return ""

def auto_discover_database(sql, cursor):
    """
    Automated database discovery algorithm utilizing the information_schema 
    to map query tables/columns to the correct schema dynamically.
    """
    tables = re.findall(r'(?i)(?:FROM|JOIN|UPDATE|INTO)\s+`?([a-zA-Z0-9_]+)`?', sql)
    columns = re.findall(r'`?([a-zA-Z0-9_]+)`?\s*(?:=|>|<|LIKE|IN|IS)', sql)
    
    tables = list(set([t for t in tables if t.upper() not in ['SELECT', 'WHERE', 'AS', 'ON', 'SET']]))
    columns = list(set(columns))
    
    if not tables:
        return None

    target_table = tables[0]
    
    try:
        cursor.execute("""
            SELECT TABLE_SCHEMA, COLUMN_NAME 
            FROM information_schema.COLUMNS 
            WHERE TABLE_NAME = %s 
            AND TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
        """, (target_table,))
        
        results = cursor.fetchall()
        if not results:
            return None
            
        db_columns_map = {}
        for row in results:
            db = row['TABLE_SCHEMA']
            col = row['COLUMN_NAME']
            if db not in db_columns_map:
                db_columns_map[db] = set()
            db_columns_map[db].add(col)
            
        if len(db_columns_map) == 1:
            return list(db_columns_map.keys())[0]
            
        best_db = None
        max_match = -1
        sql_cols_set = set(columns)
        
        for db, db_cols in db_columns_map.items():
            match_count = len(sql_cols_set.intersection(db_cols))
            if match_count > max_match:
                max_match = match_count
                best_db = db
                
        return best_db
    except:
        return None

def extract_static_features(sql):
    sql_upper = sql.upper()
    return {
        "query_length": len(sql),
        "num_joins": len(re.findall(r"\bJOIN\b", sql_upper)),
        "num_subqueries": len(re.findall(r"\(SELECT\b", sql_upper)),
        "has_wildcard": 1 if re.search(r"LIKE\s+['\"].*%.*['\"]", sql_upper) else 0,
        "has_union": 1 if re.search(r"\bUNION\b", sql_upper) else 0,
        "has_groupby_orderby": 1 if re.search(r"\b(GROUP BY|ORDER BY)\b", sql_upper) else 0,
        "has_comment": 1 if re.search(r"(--|#|/\*)", sql_upper) else 0,
    }

def extract_dynamic_features(sql, default_db=""):
    features = {
        "query_cost": 0.0, "estimated_rows": 0.0,
        "is_full_table_scan": 0, "key_used": 0,
        "explain_status": "FAILED"
    }

    dynamic_db = extract_db_from_sql(sql)
    conn = get_db_connection(dynamic_db=dynamic_db if dynamic_db else default_db)
    
    if not conn:
        features["explain_status"] = "DB_CONN_ERROR"
        return features

    try:
        with conn.cursor() as cursor:
            # If the target database is indeterminate, invoke the automated discovery algorithm
            if not dynamic_db and not default_db:
                discovered_db = auto_discover_database(sql, cursor)
                if discovered_db:
                    cursor.execute(f"USE `{discovered_db}`")
            
            cursor.execute(f"EXPLAIN FORMAT=JSON {sql}")
            explain_result = cursor.fetchone()
            
            if explain_result:
                key_name = list(explain_result.keys())[0]
                json_plan = json.loads(explain_result[key_name])
                query_block = json_plan.get("query_block", {})

                def parse_explain_node(node):
                    c, r, scan, key = 0.0, 0.0, 0, 0
                    if "table" in node:
                        tbl = node["table"]
                        cost_info = tbl.get("cost_info", {})
                        c = float(cost_info.get("read_cost", 0)) + float(cost_info.get("eval_cost", 0))
                        r = float(tbl.get("rows_examined_per_scan", 0))
                        scan = 1 if tbl.get("access_type", "").upper() == "ALL" else 0
                        key = 1 if tbl.get("key") is not None else 0
                        return c, r, scan, key
                    elif "nested_loop" in node:
                        for item in node["nested_loop"]:
                            _c, _r, _scan, _key = parse_explain_node(item)
                            c += _c
                            r += _r
                            scan = max(scan, _scan)
                            key = max(key, _key)
                    return c, r, scan, key

                c, r, scan, key = parse_explain_node(query_block)
                total_query_cost = float(query_block.get("cost_info", {}).get("query_cost", c))

                features["query_cost"] = total_query_cost
                features["estimated_rows"] = r
                features["is_full_table_scan"] = scan
                features["key_used"] = key
                features["explain_status"] = "SUCCESS"

    except pymysql.MySQLError as e:
        features["explain_status"] = f"DB_ERROR"
    except Exception as e:
        features["explain_status"] = f"SYS_ERROR"

    return features