#!/usr/bin/env python

import os
import pandas as pd
import sqlite3
import zlib
import bz2
import argparse

def decompress_database(db_path):
    """
    Decompresses a .bz2 database if it exists.
    """
    bz2_path = db_path + ".bz2"
    if os.path.exists(bz2_path):
        print(f"📂 Found compressed database. Decompressing {bz2_path}...")
        with bz2.BZ2File(bz2_path, 'rb') as bz2_file, open(db_path, 'wb') as db_file:
            db_file.write(bz2_file.read())
        print(f"✅ Decompressed database to {db_path}")

def compress_database(db_path):
    """
    Compresses the SQLite database using bzip2 and removes the uncompressed file.
    """
    bz2_path = db_path + ".bz2"
    
    with open(db_path, 'rb') as db_file, bz2.BZ2File(bz2_path, 'wb') as bz2_file:
        bz2_file.write(db_file.read())

    os.remove(db_path)  # Remove original uncompressed database
    print(f"📦 Database compressed to {bz2_path} and original removed.")

def compress_column(df):
    """
    Compress all columns that contain string data (likely large text or sequences).
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is of type 'object' (string in pandas)
            print(f"💾 Compressing column: {col}")
            df[col] = df[col].apply(lambda x: zlib.compress(x.encode('utf-8')) if isinstance(x, str) else x)
    return df

def detect_column_type(column):
    """
    Detect the column type for SQLite storage. This will map pandas data types to SQLite types.
    """
    if column.dtype == 'object':  # String columns
        return "BLOB"
    elif column.dtype == 'int64':  # Integer columns
        return "INTEGER"
    elif column.dtype == 'float64':  # Float columns
        return "REAL"
    else:
        return "TEXT"  # Default to TEXT for other cases

def store_table(conn, cursor, table_name, df, species, condition, isolate, metadata_file):
    """
    Stores a Pandas DataFrame into the SQLite database efficiently.
    Automatically compresses string columns and updates metadata file.
    """
    if 'isolate' not in df.columns:
        df['isolate'] = isolate  
    df['species'] = species  
    df['condition'] = condition  

    cursor.execute("PRAGMA journal_mode=WAL;")  
    cursor.execute("PRAGMA synchronous=NORMAL;")  
    cursor.execute("PRAGMA cache_size=-100000;")  
    cursor.execute("PRAGMA temp_store=MEMORY;")  

    df = df.drop_duplicates()
    df = compress_column(df)  # Apply zlib compression to text columns

    columns = [f'"{col}" {detect_column_type(df[col])}' for col in df.columns]
    create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(columns)})'
    cursor.execute(create_table_sql)

    df.to_sql(table_name, conn, if_exists="append", index=False, chunksize=500)
    print(f"✅ Stored {table_name} successfully!")

    add_metadata_entry(metadata_file, table_name, species, condition, isolate)

    cursor.execute("VACUUM;")  
    conn.commit()
    print(f"🔍 Optimized {table_name} storage")

def add_metadata_entry(metadata_file, table_name, species, condition, isolate):
    """
    Adds an entry to the metadata file that tracks the database tables.
    """
    if not os.path.exists(metadata_file):
        metadata_df = pd.DataFrame(columns=["id", "table_name", "species", "condition", "isolate"])
        metadata_df.to_csv(metadata_file, sep="\t", index=False)

    metadata_df = pd.read_csv(metadata_file, sep="\t")
    new_entry = {
        "id": len(metadata_df) + 1,
        "table_name": table_name,
        "species": species,
        "condition": condition,
        "isolate": isolate
    }

    # Use pd.concat instead of deprecated .append
    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_entry])], ignore_index=True)

    metadata_df.to_csv(metadata_file, sep="\t", index=False)
    print(f"📋 Added metadata for table: {table_name}")

def process_file(conn, cursor, filepath, species, condition, isolate, metadata_file):
    """
    Reads a CSV/TSV file and stores it efficiently.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File {filepath} not found!")
        return
    
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".tsv") or filepath.endswith(".txt"):
        df = pd.read_csv(filepath, sep="\t")
    else:
        print(f"❌ Unsupported file format: {filepath}")
        return
    
    table_name = f"{species}_{condition}_{isolate}"
    store_table(conn, cursor, table_name, df, species, condition, isolate, metadata_file)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiently store tables in SQLite and compress final database with bzip2.")
    parser.add_argument("files", nargs="+", help="Path to one or more CSV/TSV files.")
    parser.add_argument("--species", required=True, help="Species name for tracking")
    parser.add_argument("--condition", required=True, help="Condition for tracking")
    parser.add_argument("--isolate", required=True, help="Isolate name for tracking")
    parser.add_argument("--db_dir", default=os.getcwd(), help="Directory to store the database and metadata file")
    
    args = parser.parse_args()
    
    os.makedirs(args.db_dir, exist_ok=True)
    
    db_path = os.path.join(args.db_dir, "efficient_data_storage.db")
    metadata_file = os.path.join(args.db_dir, "metadata_tracking.tsv")

    decompress_database(db_path)  # Unzip database if exists
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for file in args.files:
        process_file(conn, cursor, file, args.species, args.condition, args.isolate, metadata_file)
    
    conn.close()
    print(f"📁 Database stored at: {db_path}")

    compress_database(db_path)  # Recompress database after processing

    print(f"📋 Metadata tracked in: {metadata_file}")
    print("📁 Database connection closed.")
