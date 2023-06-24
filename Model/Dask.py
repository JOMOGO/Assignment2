import dask.dataframe as dd
from dask_ml.feature_extraction.text import HashingVectorizer
from dask_ml.linear_model import LogisticRegression
from dask.distributed import Client
from nltk.corpus import stopwords
from dask_ml.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from Database.db_connector import load_data_from_mongodb

nltk.download('stopwords')

# Connect to Dask client
client = Client()

try:
    # Load data from MongoDB
    df = load_data_from_mongodb()

    # Convert pandas dataframe to dask dataframe
    ddf = dd.from_pandas(df, npartitions=2)

    # Define features and target
    X = ddf['Positive_Review'].str.lower().str.replace('[^\w\s]', '').to_frame()
    y = (ddf['Reviewer_Score'] > 5).to_frame()  # Let's say a score > 5 is positive sentiment

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a vectorizer
    vectorizer = HashingVectorizer(stop_words=stopwords.words('english'))

    # Vectorize the text reviews
    X_train = vectorizer.fit_transform(X_train.values.compute())
    X_test = vectorizer.transform(X_test.values.compute())

    # Initialize a logistic regression model
    model = LogisticRegression()

    # Fit the model
    model.fit(X_train, y_train.values.compute())

    # Predict the test set results
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test.values.compute(), y_pred)

    print(f'Accuracy: {accuracy}')
    client.close()
finally:
    client.close()
