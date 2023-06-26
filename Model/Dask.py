import numpy as np
import pandas as pd
from dask_ml.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dask_ml.wrappers import Incremental
from dask.distributed import Client
from nltk.corpus import stopwords
from dask_ml.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from Database.db_connector import load_data_from_mongodb
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional
from keras.utils.np_utils import to_categorical

# Create a list of models to evaluate
# models = [
#     {"name": "SGDClassifier", "model": SGDClassifier()},
#     {"name": "LogisticRegression", "model": LogisticRegression()}
# ]

if __name__ == '__main__':
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
        df['Sentiment'] = df['Reviewer_Score'].apply(lambda x: 'positive' if x > 5 else 'negative')
        y = df['Sentiment']

        # Convert X and y to list
        X = X.astype(str).tolist()
        y = y.tolist()

        # Tokenizing text
        tokenizer = Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(X)
        joblib.dump(tokenizer, '../Model_results/tokenizer.pkl')

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Padding sequences
        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=300)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=300)

        # Initialize a vectorizer
        vectorizer = HashingVectorizer(stop_words=stopwords.words('english'))
        X_train = vectorizer.fit_transform(X_train)
        joblib.dump(vectorizer, '../Model_results/vectorizer.pkl')
        X_test = vectorizer.transform(X_test)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # for m in models:
        #     model_file = f"../Model_results/{m['name']}_model.pkl"
        #     if os.path.exists(model_file):
        #         print(f"Loading {m['name']} from disk...")
        #         model = joblib.load(model_file)
        #     else:
        #         print(f"Training {m['name']}...")
        #         if hasattr(m['model'], 'partial_fit'):
        #             model = Incremental(m['model'])
        #             model.fit(X_train, y_train_encoded, classes=np.unique(y_train_encoded))
        #         else:
        #             model = m['model']
        #             model.fit(X_train.toarray(), y_train_encoded)  # Convert to dense array
        #         joblib.dump(model, model_file)
        #
        #     # Predict the test data
        #     if hasattr(m['model'], 'partial_fit'):
        #         y_pred = model.predict(X_test)
        #     else:
        #         y_pred = model.predict(X_test.toarray())  # Convert to dense array
        #
        #     print(f"Accuracy of {m['name']}: ", accuracy_score(y_test_encoded, y_pred))
        #     print(f"Classification report of {m['name']}:")
        #     print(classification_report(y_test_encoded, y_pred, zero_division=1))

        # Convert target variable to categorical
        y_train_cat = to_categorical(y_train_encoded)
        y_test_cat = to_categorical(y_test_encoded)

        # Define RNN model
        model_rnn = Sequential()
        model_rnn.add(Embedding(2000, 50, input_length=300))
        model_rnn.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))
        model_rnn.add(Dense(2, activation='sigmoid'))

        # Compile the model
        model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        model_rnn.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the RNN model
        model_rnn.save('../Model_results/rnn_model.h5')

        # Evaluate the RNN model
        scores = model_rnn.evaluate(X_test_seq, y_test_cat)
        print("RNN Accuracy: ", scores[1])

        # Define CNN model
        model_cnn = Sequential()
        model_cnn.add(Embedding(2000, 50, input_length=300))
        model_cnn.add(Conv1D(128, 5, activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(2, activation='sigmoid'))

        # Compile the model
        model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        model_cnn.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the CNN model
        model_cnn.save('../Model_results/cnn_model.h5')

        # Evaluate the CNN model
        scores = model_cnn.evaluate(X_test_seq, y_test_cat)
        print("CNN Accuracy: ", scores[1])

        # Define Bi-LSTM model
        model_bilstm = Sequential()
        model_bilstm.add(Embedding(2000, 50, input_length=300))
        model_bilstm.add(Bidirectional(LSTM(25, dropout=0.2, recurrent_dropout=0.2)))
        model_bilstm.add(Dense(2, activation='sigmoid'))

        # Compile the model
        model_bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        model_bilstm.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the Bi-LSTM model
        model_bilstm.save('../Model_results/bilstm_model.h5')

        # Evaluate the Bi-LSTM model
        scores = model_bilstm.evaluate(X_test_seq, y_test_cat)
        print("Bi-LSTM Accuracy: ", scores[1])


        client.close()
    finally:
        client.close()
