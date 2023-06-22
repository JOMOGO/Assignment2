from pymongo import MongoClient
import pandas as pd

# Connect to mongodb
try:
    client = MongoClient("localhost:27017")
    db = client.hotel_db
except Exception as e:
    print("Could not connect to MongoDB: ", e)

# Load csv and put it in MongoDB
try:
    hotel_review_df = pd.read_csv(filepath_or_buffer='Hotel_Reviews.csv', engine='python', on_bad_lines='skip')
    # Perform any necessary data cleaning here

    db.hotel_reviews.insert_many(hotel_review_df.to_dict('records'))
except Exception as e:
    print("Could not load CSV or insert data into MongoDB: ", e)

# Close the connection
client.close()
