import pandas as pd
from pymongo import MongoClient

def load_data_from_mongodb():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["hotel_db"]
    collection = db["hotel_reviews"]

    # Load data from MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)

    return df, collection