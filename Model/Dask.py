import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.feature_extraction.text import HashingVectorizer
from dask.distributed import Client
from nltk.corpus import stopwords
from dask_ml.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from Database.db_connector import load_data_from_mongodb
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional
from keras.utils.np_utils import to_categorical
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import re


def select_review(row):
    if row['Reviewer_Score'] > 7 and row['Positive_Review'] != '':
        text = row['Positive_Review']
    elif row['Reviewer_Score'] < 5 and row['Negative_Review'] != '':
        text = row['Negative_Review']
    else:
        return np.nan

    processed_text = re.sub(
        r'\b(not|no|never|neither|nothing|none|no one|nobody|nowhere|nor|barely|hardly|scarcely|seldom|rarely)\s+(\w+)\b',
        r'\1_\2', text)
    return processed_text

def plot_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    x = labels
    y = labels
    fig = ff.create_annotated_heatmap(matrix, x=x, y=y, colorscale='Viridis')
    fig.update_layout(
        title_text='Confusion Matrix',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label')
    )
    return fig

def plot_roc_auc(y_true, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    trace0 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc)
    trace1 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash'))

    layout = go.Layout(title='Receiver Operating Characteristic for ' + model_name,
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=[trace0, trace1], layout=layout)
    return fig

def map_sentiment(df):
    return df['Reviewer_Score'].apply(lambda x: 'positive' if x > 7 else ('negative' if x < 5 else np.nan))

def map_select_review(df):
    return df.apply(select_review, axis=1)

def train_model(X_train_seq, y_train_cat, model_type):
    model = Sequential()
    model.add(Embedding(2000, 50, input_length=300))
    if model_type == 'RNN':
        model.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='sigmoid'))
    elif model_type == 'CNN':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='sigmoid'))
    elif model_type == 'Bi-LSTM':
        model.add(Bidirectional(LSTM(25, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)
    return model, history


if __name__ == '__main__':
    client = Client()

    try:
        df, collection = load_data_from_mongodb()

        df['Positive_Review'] = df['Positive_Review'].astype(str)
        df['Negative_Review'] = df['Negative_Review'].astype(str)

        # convert pandas DataFrame to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=2)

        ddf['Review'] = ddf.map_partitions(map_select_review, meta=('Review', 'object')).compute()
        ddf = ddf.dropna(subset=['Review'])

        ddf['Sentiment'] = ddf.map_partitions(map_sentiment, meta=('Sentiment', 'object')).compute()
        ddf = ddf.dropna(subset=['Sentiment'])

        # Drop NaN values from 'Review' and 'Sentiment' columns
        ddf = ddf.dropna(subset=['Review', 'Sentiment'])

        # Count the occurrences of positive and negative reviews
        positive_count = (ddf['Sentiment'] == 'positive').sum().compute()
        negative_count = (ddf['Sentiment'] == 'negative').sum().compute()

        # Determine the minimum count between positive and negative reviews
        min_count = min(positive_count, negative_count)

        # Sample the minimum count of positive and negative reviews
        positive_reviews = ddf[ddf['Sentiment'] == 'positive'].sample(frac=min_count / positive_count, random_state=42)
        negative_reviews = ddf[ddf['Sentiment'] == 'negative'].sample(frac=min_count / negative_count, random_state=42)

        # Concatenate the positive and negative reviews
        balanced_df = dd.concat([positive_reviews, negative_reviews])

        # Shuffle the rows in the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Use the balanced dataset for training and testing
        X = balanced_df['Review']
        y = balanced_df['Sentiment']

        # Convert X and y to pandas Series by calling .compute(), then convert to list
        X = X.compute().tolist()
        y = y.compute().tolist()

        tokenizer = Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(X)
        joblib.dump(tokenizer, '../Model_results/tokenizer.pkl')

        ros = RandomOverSampler(random_state=42)

        # Splitting data into training and testing using Dask
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=300)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=300)

        X_resampled, y_resampled = ros.fit_resample(pd.DataFrame(X_train_seq), y_train)

        X_resampled = [str(doc) for doc in X_resampled[0]]  # Convert X_resampled to a list of strings
        y_resampled = pd.Series(y_resampled)  # Transform resampled y into a pandas Series

        vectorizer = HashingVectorizer(stop_words=stopwords.words('english'))
        X_train_transformed = vectorizer.fit_transform(X_resampled)
        X_test_transformed = vectorizer.transform(X_test)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_resampled)
        y_test_encoded = le.transform(y_test)

        # Convert target variable to categorical
        y_train_cat = to_categorical(y_train_encoded)
        y_test_cat = to_categorical(y_test_encoded)

        model_rnn, history_rnn = train_model(X_train_seq, y_train_cat, 'RNN')
        model_cnn, history_cnn = train_model(X_train_seq, y_train_cat, 'CNN')
        model_bilstm, history_bilstm = train_model(X_train_seq, y_train_cat, 'Bi-LSTM')

        rnn_preds = model_rnn.predict(X_test_seq)
        cnn_preds = model_cnn.predict(X_test_seq)
        bilstm_preds = model_bilstm.predict(X_test_seq)

        # Plot and export figures
        cm_fig_rnn = plot_confusion_matrix(y_test_cat, rnn_preds, ['positive', 'negative'])
        roc_fig_rnn = plot_roc_auc(y_test_cat, rnn_preds, 'RNN')

        cm_fig_cnn = plot_confusion_matrix(y_test_cat, cnn_preds, ['positive', 'negative'])
        roc_fig_cnn = plot_roc_auc(y_test_cat, cnn_preds, 'CNN')

        cm_fig_bilstm = plot_confusion_matrix(y_test_cat, bilstm_preds, ['positive', 'negative'])
        roc_fig_bilstm = plot_roc_auc(y_test_cat, bilstm_preds, 'Bi-LSTM')

        # Save the figures as .json
        pio.write_json(cm_fig_rnn, '../Model_results/cm_fig_rnn.json')
        pio.write_json(roc_fig_rnn, '../Model_results/roc_fig_rnn.json')

        pio.write_json(cm_fig_cnn, '../Model_results/cm_fig_cnn.json')
        pio.write_json(roc_fig_cnn, '../Model_results/roc_fig_cnn.json')

        pio.write_json(cm_fig_bilstm, '../Model_results/cm_fig_bilstm.json')
        pio.write_json(roc_fig_bilstm, '../Model_results/roc_fig_bilstm.json')

        client.close()
    finally:
        client.close()
