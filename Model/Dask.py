import numpy as np
from dask_ml.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from dask_ml.wrappers import Incremental
from dask.distributed import Client
from nltk.corpus import stopwords
from dask_ml.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from sklearn.preprocessing import LabelEncoder
from Database.db_connector import load_data_from_mongodb

if __name__ == '__main__':

    nltk.download('stopwords', quiet=True)

    # Connect to Dask client
    client = Client()

    try:
        # Load data from MongoDB
        df = load_data_from_mongodb()

        # Convert 'Positive_Review' and 'Negative_Review' to lists of strings
        df['Positive_Review'] = df['Positive_Review'].astype(str)
        df['Negative_Review'] = df['Negative_Review'].astype(str)

        # Define X and y
        X = df['Positive_Review'] + df['Negative_Review']
        y = df['Reviewer_Score']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert X_train and X_test to lists of strings
        X_train = X_train.astype(str).tolist()
        X_test = X_test.astype(str).tolist()

        # Initialize a vectorizer
        vectorizer = HashingVectorizer(stop_words=stopwords.words('english'))

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Create a scikit-learn SGDClassifier model
        sgd = SGDClassifier()

        # Wrap the model with Dask Incremental
        model = Incremental(sgd)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        model.fit(X_train, y_train_encoded, classes=np.unique(y_train_encoded))

        # Predict the test data
        y_pred = model.predict(X_test)

        y_test_encoded = le.transform(y_test)
        print(accuracy_score(y_test_encoded, y_pred))

        client.close()
    finally:
        client.close()
