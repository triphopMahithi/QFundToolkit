import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

def generate_cache_filename(base_name="yf_cache", filetype="pkl", period="1y", interval="1d", cache_dir="cache"):
    """
    Generate standardized cache file names.
    Example: yf_cache_1y_1d_2025-08-01.pkl
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{base_name}_{period}_{interval}_{timestamp}.{filetype}"
    return os.path.join(cache_dir, filename)

def save_to_pickle(df, period="1y", interval="1d", cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    filename = generate_cache_filename(filetype="pkl", period=period, interval=interval, cache_dir=cache_dir)
    df.to_pickle(filename)
    print(f"✅ Cached to Pickle file: {filename}")
    return filename

def save_to_csv(df, period="1y", interval="1d", cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    filename = generate_cache_filename(filetype="csv", period=period, interval=interval, cache_dir=cache_dir)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
    df.to_csv(filename, index=True)
    print(f"✅ Cached to CSV file: {filename}")
    return filename

def load_from_pickle(filepath):
    return pd.read_pickle(filepath)

# FIX: MongoDB ไม่อนุญาตให้ใช้ key ที่เป็นตัวเลขล้วน 
def save_to_mongo(df, mongo_uri, db_name, collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns.values]
    
    df_reset = df.reset_index()
    records = df_reset.to_dict(orient='records')
    for record in records:
        record['Date'] = str(record['Date'])
        query = {"Date": record['Date']}
        collection.update_one(query, {"$set": record}, upsert=True)

    client.close()
    print("✅ Cached to MongoDB")