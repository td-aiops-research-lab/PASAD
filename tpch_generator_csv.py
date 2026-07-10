"""
File: tpch_generator_csv.py
Description: 
    A Python utility designed to synthesize data 
    compliant with the TPC-H benchmark schema. This script serves as an 
    alternative to the standard DBGEN utility for rapid testing environments.
    
Execution Protocol:
    1. Execute the script via the Python interpreter: python tpch_generator_csv.py
    2. Await the generation of the target .csv artifacts (e.g., customer.csv, orders.csv, lineitem.csv).
    3. Utilize the MySQL LOAD DATA INFILE command to ingest the synthesized datasets into the target database.
"""

import csv
import random
import os
from datetime import date, timedelta

# ==========================================
# DATASET VOLUME CONFIGURATION (Scale configurations to simulate high load performance testing)
# ==========================================
SCALE_FACTOR = 100  # Increment this scalar (e.g., 10, 50, 100) to synthesize millions of records
NUM_REGION = 5
NUM_NATION = 25
NUM_PART = 1000 * SCALE_FACTOR
NUM_SUPPLIER = 100 * SCALE_FACTOR
NUM_CUSTOMER = 1500 * SCALE_FACTOR
NUM_ORDERS = 15000 * SCALE_FACTOR
NUM_LINEITEM = 60000 * SCALE_FACTOR

# Stochastic constants formulated to guarantee non-empty result sets during query execution testing
MKT_SEGMENTS = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
PART_TYPES = ['COPPER', 'BRASS', 'STEEL', 'TIN', 'PLASTIC', 'COPPER NICKEL']
ORDER_STATUS = ['O', 'F', 'P']
ORDER_PRIORITY = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']
RETURN_FLAG = ['R', 'A', 'N']
LINE_STATUS = ['O', 'F']

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

def generate_csv(filename, columns, data_generator, num_rows):
    print(f"[*] Synthesizing artifact: {filename} ({num_rows} records)...")
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Retain headers for structural readability. During MySQL ingestion, 
        # employ the 'IGNORE 1 ROWS' directive to bypass header parsing.
        writer.writerow(columns)
        
        for _ in range(num_rows):
            writer.writerow(next(data_generator))
    print(f"[*] Successfully finalized artifact: {filename}")

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
        ship_date = random_date(1995, 1998)
        commit_date = random_date(1995, 1998)
        receipt_date = random_date(1995, 1998)
        yield [random.randint(1, NUM_ORDERS), random.randint(1, NUM_PART), random.randint(1, NUM_SUPPLIER),
               random.randint(1, 5), random.randint(1, 100), round(random.uniform(10.0, 1000.0), 2),
               round(random.uniform(0.0, 0.1), 2), round(random.uniform(0.0, 0.08), 2),
               random.choice(RETURN_FLAG), random.choice(LINE_STATUS), ship_date, commit_date, receipt_date,
               "DELIVER IN PERSON", "TRUCK", random_string(20)]

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    print("[INFO] Initiating TPC-H compliant mock data synthesis pipeline...")
    
    generate_csv('region.csv', ['r_regionkey', 'r_name', 'r_comment'], gen_region(), NUM_REGION)
    generate_csv('nation.csv', ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], gen_nation(), NUM_NATION)
    generate_csv('part.csv', ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'], gen_part(), NUM_PART)
    generate_csv('supplier.csv', ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'], gen_supplier(), NUM_SUPPLIER)
    generate_csv('customer.csv', ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'], gen_customer(), NUM_CUSTOMER)
    generate_csv('orders.csv', ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'], gen_orders(), NUM_ORDERS)
    generate_csv('lineitem.csv', ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], gen_lineitem(), NUM_LINEITEM)

    print("\n[SUCCESS] Synthesis pipeline completed.")
    print("[INFO] Execute the following SQL directive within the MySQL environment for data ingestion (adjust the file path accordingly):")
    print("""
    LOAD DATA LOCAL INFILE 'customer.csv' 
    INTO TABLE customer 
    FIELDS TERMINATED BY ',' 
    ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE 1 ROWS;
    """)