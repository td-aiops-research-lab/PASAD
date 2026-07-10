"""
File: tpch_generator_sql.py
Description: 
    A Python script designed to synthesize the TPC-H 
    benchmark database into a single .sql file. 
    It leverages batched inserts to optimize insertion speed and toggles 
    Foreign Key Checks to prevent referential integrity errors during bulk imports.
"""

import random
from datetime import date, timedelta
import time

# ==========================================
# DATASET VOLUME CONFIGURATION
# ==========================================
SCALE_FACTOR = 1000  # Note: The resulting SQL artifact will be exceptionally large at higher scale factors.
NUM_REGION = 5
NUM_NATION = 25
NUM_PART = 1000 * SCALE_FACTOR
NUM_SUPPLIER = 100 * SCALE_FACTOR
NUM_CUSTOMER = 1500 * SCALE_FACTOR
NUM_ORDERS = 15000 * SCALE_FACTOR
NUM_LINEITEM = 60000 * SCALE_FACTOR

BATCH_SIZE = 2000 # Record count per INSERT statement (Optimized for maximum ingestion throughput)
OUTPUT_FILE = 'tpch_full_database.sql'

# Stochastic Constants
MKT_SEGMENTS = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
PART_TYPES = ['COPPER', 'BRASS', 'STEEL', 'TIN', 'PLASTIC', 'COPPER NICKEL']
ORDER_STATUS = ['O', 'F', 'P']
ORDER_PRIORITY = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']
RETURN_FLAG = ['R', 'A', 'N']
LINE_STATUS = ['O', 'F']

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def random_date(start_year=1992, end_year=1998):
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return (start_date + timedelta(days=random_number_of_days)).strftime('%Y-%m-%d')

def random_string(length=10):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return ''.join(random.choice(letters) for _ in range(length))

def format_sql_value(val):
    """Formats scalar values for compliant SQL insertion."""
    if isinstance(val, str):
        # Escape single quotes to prevent SQL syntax violations and injection vulnerabilities
        val = val.replace("'", "''")
        return f"'{val}'"
    elif isinstance(val, (int, float)):
        return str(val)
    elif val is None:
        return "NULL"
    return f"'{str(val)}'"

def write_sql_batch(file, table_name, columns, data_generator, num_rows):
    """Executes chunked data serialization (Batch Insertion) into the target SQL artifact."""
    print(f"[*] Synthesizing dataset for relation: {table_name} ({num_rows} records)...")
    columns_str = ", ".join(columns)
    
    count = 0
    batch_values = []
    
    for row in data_generator:
        # Format individual attributes within the tuple
        formatted_row = [format_sql_value(v) for v in row]
        batch_values.append(f"({', '.join(formatted_row)})")
        count += 1
        
        # Trigger write operation upon reaching batch capacity or terminal record
        if count % BATCH_SIZE == 0 or count == num_rows:
            insert_stmt = f"INSERT INTO {table_name} ({columns_str}) VALUES \n"
            insert_stmt += ",\n".join(batch_values) + ";\n\n"
            file.write(insert_stmt)
            batch_values = [] # Reinitialize batch buffer
            
    print(f"[*] Successfully finalized relation: {table_name}")

# ==========================================
# DATA SYNTHESIS GENERATOR FUNCTIONS
# ==========================================
def gen_region():
    for i in range(1, NUM_REGION + 1):
        yield [i, f"Region_{i}", random_string(20)]

def gen_nation():
    for i in range(1, NUM_NATION + 1):
        yield [i, f"Nation_{i}", random.randint(1, NUM_REGION), random_string(20)]

def gen_part():
    for i in range(1, NUM_PART + 1):
        brand_id = random.randint(1, 50)
        yield [i, f"Part_{i}_{random_string(5)}", f"Manufacturer_{random.randint(1,5)}", 
               f"Brand#{brand_id}", random.choice(PART_TYPES), random.randint(1, 50), 
               f"Container_{random.randint(1,10)}", round(random.uniform(10.0, 1000.0), 2), random_string(10)]

def gen_supplier():
    for i in range(1, NUM_SUPPLIER + 1):
        yield [i, f"Supplier_{i}", random_string(15), random.randint(1, NUM_NATION), 
               f"123-456-{random.randint(1000, 9999)}", round(random.uniform(-100.0, 10000.0), 2), random_string(20)]

def gen_customer():
    for i in range(1, NUM_CUSTOMER + 1):
        yield [i, f"Customer_{i}", random_string(15), random.randint(1, NUM_NATION),
               f"987-654-{random.randint(1000, 9999)}", round(random.uniform(-100.0, 10000.0), 2),
               random.choice(MKT_SEGMENTS), random_string(20)]

def gen_orders():
    for i in range(1, NUM_ORDERS + 1):
        o_date = random_date()
        yield [i, random.randint(1, NUM_CUSTOMER), random.choice(ORDER_STATUS),
               round(random.uniform(100.0, 100000.0), 2), o_date,
               random.choice(ORDER_PRIORITY), f"Clerk_{random.randint(1, 100)}", 0, random_string(20)]

def gen_lineitem():
    for i in range(1, NUM_LINEITEM + 1):
        yield [random.randint(1, NUM_ORDERS), random.randint(1, NUM_PART), random.randint(1, NUM_SUPPLIER),
               random.randint(1, 5), random.randint(1, 100), round(random.uniform(10.0, 1000.0), 2),
               round(random.uniform(0.0, 0.1), 2), round(random.uniform(0.0, 0.08), 2),
               random.choice(RETURN_FLAG), random.choice(LINE_STATUS), random_date(1995, 1998), 
               random_date(1995, 1998), random_date(1995, 1998),
               "DELIVER IN PERSON", "TRUCK", random_string(20)]

# ==========================================
# SCHEMA DEFINITION (DDL) AND EXECUTION ROUTINE
# ==========================================
DDL_SCHEMA = """
-- ========================================================
-- TPC-H DATABASE SCHEMA & CONSTRAINTS
-- ========================================================
SET FOREIGN_KEY_CHECKS = 0; -- DISABLE REFERENTIAL INTEGRITY CHECKS FOR ACCELERATED INGESTION

DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;

CREATE TABLE region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name CHAR(25),
    r_comment VARCHAR(152)
);

CREATE TABLE nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name CHAR(25),
    n_regionkey INTEGER,
    n_comment VARCHAR(152),
    FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
);

CREATE TABLE part (
    p_partkey INTEGER PRIMARY KEY,
    p_name VARCHAR(55),
    p_mfgr CHAR(25),
    p_brand CHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container CHAR(50),
    p_retailprice DECIMAL(15,2),
    p_comment VARCHAR(23)
);

CREATE TABLE supplier (
    s_suppkey INTEGER PRIMARY KEY,
    s_name CHAR(25),
    s_address VARCHAR(40),
    s_nationkey INTEGER,
    s_phone CHAR(15),
    s_acctbal DECIMAL(15,2),
    s_comment VARCHAR(101),
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE customer (
    c_custkey INTEGER PRIMARY KEY,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER,
    c_phone CHAR(15),
    c_acctbal DECIMAL(15,2),
    c_mktsegment CHAR(10),
    c_comment VARCHAR(117),
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE orders (
    o_orderkey INTEGER PRIMARY KEY,
    o_custkey INTEGER,
    o_orderstatus CHAR(1),
    o_totalprice DECIMAL(15,2),
    o_orderdate DATE,
    o_orderpriority CHAR(15),
    o_clerk CHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79),
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

CREATE TABLE lineitem (
    l_orderkey INTEGER,
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount DECIMAL(15,2),
    l_tax DECIMAL(15,2),
    l_returnflag CHAR(1),
    l_linestatus CHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct CHAR(25),
    l_shipmode CHAR(10),
    l_comment VARCHAR(44),
    -- Composite Primary Key
    PRIMARY KEY (l_orderkey, l_linenumber),
    -- Foreign Key Definitions
    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
    FOREIGN KEY (l_partkey) REFERENCES part(p_partkey),
    FOREIGN KEY (l_suppkey) REFERENCES supplier(s_suppkey)
);

-- ========================================================
-- INITIATE DATA INSERTION (BATCHED)
-- ========================================================
"""

if __name__ == "__main__":
    start_time = time.time()
    print(f"[INFO] Initiating synthesis of consolidated database artifact ({OUTPUT_FILE})...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Prepend table instantiation DDL and disable foreign key checks
        f.write(DDL_SCHEMA)
        
        # Enforce rigorous PARENT-CHILD hierarchy to prevent referential integrity violations
        write_sql_batch(f, 'region', ['r_regionkey', 'r_name', 'r_comment'], gen_region(), NUM_REGION)
        write_sql_batch(f, 'nation', ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], gen_nation(), NUM_NATION)
        write_sql_batch(f, 'part', ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'], gen_part(), NUM_PART)
        write_sql_batch(f, 'supplier', ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'], gen_supplier(), NUM_SUPPLIER)
        write_sql_batch(f, 'customer', ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'], gen_customer(), NUM_CUSTOMER)
        write_sql_batch(f, 'orders', ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'], gen_orders(), NUM_ORDERS)
        write_sql_batch(f, 'lineitem', ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], gen_lineitem(), NUM_LINEITEM)
        
        # Restore referential integrity constraints post-ingestion
        f.write("\nSET FOREIGN_KEY_CHECKS = 1;\n")
        f.write("-- INGESTION PROTOCOL COMPLETE --\n")

    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Artifact successfully generated: {OUTPUT_FILE} (Execution time: {elapsed_time:.2f}s)")
    print("[INFO] Execute the following command within the MySQL terminal to initialize the database:")
    print(f"mysql -u username -p database_name < {OUTPUT_FILE}")