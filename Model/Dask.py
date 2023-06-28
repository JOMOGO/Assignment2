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

def select_review(row):
    if row['Reviewer_Score'] > 6:
        return row['Positive_Review']
    elif row['Reviewer_Score'] < 4:
        return row['Negative_Review']
    else:
        return np.nan

def plot_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    x = labels
    y = labels[::-1]

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

def plot_loss_progression(history, model_name):
    trace0 = go.Scatter(x=history.epoch, y=history.history['loss'], mode='lines', name='train')
    trace1 = go.Scatter(x=history.epoch, y=history.history['val_loss'], mode='lines', name='test')

    layout = go.Layout(title='Loss progression for ' + model_name)

    fig = go.Figure(data=[trace0, trace1], layout=layout)

    return fig

if __name__ == '__main__':
    client = Client()

    try:
        df, collection = load_data_from_mongodb()

        df['Positive_Review'] = df['Positive_Review'].astype(str)
        df['Negative_Review'] = df['Negative_Review'].astype(str)

        df['Review'] = df.apply(select_review, axis=1)
        df = df.dropna(subset=['Review'])

        df['Sentiment'] = df['Reviewer_Score'].apply(lambda x: 'positive' if x > 6 else ('negative' if x < 4 else np.nan))
        df = df.dropna(subset=['Sentiment'])

        X = df['Review']
        y = df['Sentiment']

        X = X.astype(str).tolist()
        y = y.tolist()

        tokenizer = Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(X)
        joblib.dump(tokenizer, '../Model_results/tokenizer.pkl')

        ros = RandomOverSampler(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=300)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=300)

        X_resampled, y_resampled = ros.fit_resample(pd.DataFrame(X_train_seq), y_train)

        X_resampled = [str(doc) for doc in X_resampled[0]]  # Convert X_resampled to a list of strings
        y_resampled = pd.Series(y_resampled)  # Transform resampled y into a pandas Series

        vectorizer = HashingVectorizer(stop_words=stopwords.words('english'))
        X_train_transformed = vectorizer.fit_transform(X_resampled)
        joblib.dump(vectorizer, '../Model_results/vectorizer.pkl')
        X_test_transformed = vectorizer.transform(X_test)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_resampled)
        y_test_encoded = le.transform(y_test)

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
        history_rnn = model_rnn.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the RNN model
        model_rnn.save('../Model_results/rnn_model.h5')

        # Evaluate the RNN model
        scores = model_rnn.evaluate(X_test_seq, y_test_cat)
        print("RNN Accuracy: ", scores[1])

        # Predict the results
        rnn_preds = model_rnn.predict(X_test_seq)

        # Define CNN model
        model_cnn = Sequential()
        model_cnn.add(Embedding(2000, 50, input_length=300))
        model_cnn.add(Conv1D(128, 5, activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(2, activation='sigmoid'))

        # Compile the model
        model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        history_cnn = model_cnn.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the CNN model
        model_cnn.save('../Model_results/cnn_model.h5')

        # Evaluate the CNN model
        scores = model_cnn.evaluate(X_test_seq, y_test_cat)
        print("CNN Accuracy: ", scores[1])

        # Predict the results
        cnn_preds = model_cnn.predict(X_test_seq)

        # Define Bi-LSTM model
        model_bilstm = Sequential()
        model_bilstm.add(Embedding(2000, 50, input_length=300))
        model_bilstm.add(Bidirectional(LSTM(25, dropout=0.2, recurrent_dropout=0.2)))
        model_bilstm.add(Dense(2, activation='sigmoid'))

        # Compile the model
        model_bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the model
        history_bilstm = model_bilstm.fit(X_train_seq, y_train_cat, validation_split=0.2, epochs=1)

        # Save the Bi-LSTM model
        model_bilstm.save('../Model_results/bilstm_model.h5')

        # Evaluate the Bi-LSTM model
        scores = model_bilstm.evaluate(X_test_seq, y_test_cat)
        print("Bi-LSTM Accuracy: ", scores[1])

        # Predict the results
        bilstm_preds = model_bilstm.predict(X_test_seq)

        # Plot and export figures
        cm_fig_rnn = plot_confusion_matrix(y_test_cat, rnn_preds, ['positive', 'negative'])
        roc_fig_rnn = plot_roc_auc(y_test_cat, rnn_preds, 'RNN')
        lp_fig_rnn = plot_loss_progression(history_rnn, 'RNN')

        cm_fig_cnn = plot_confusion_matrix(y_test_cat, cnn_preds, ['positive', 'negative'])
        roc_fig_cnn = plot_roc_auc(y_test_cat, cnn_preds, 'CNN')
        lp_fig_cnn = plot_loss_progression(history_cnn, 'CNN')

        cm_fig_bilstm = plot_confusion_matrix(y_test_cat, bilstm_preds, ['positive', 'negative'])
        roc_fig_bilstm = plot_roc_auc(y_test_cat, bilstm_preds, 'Bi-LSTM')
        lp_fig_bilstm = plot_loss_progression(history_bilstm, 'Bi-LSTM')

        # Save the figures as .json
        pio.write_json(cm_fig_rnn, '../Model_results/cm_fig_rnn.json')
        pio.write_json(roc_fig_rnn, '../Model_results/roc_fig_rnn.json')
        pio.write_json(lp_fig_rnn, '../Model_results/lp_fig_rnn.json')

        pio.write_json(cm_fig_cnn, '../Model_results/cm_fig_cnn.json')
        pio.write_json(roc_fig_cnn, '../Model_results/roc_fig_cnn.json')
        pio.write_json(lp_fig_cnn, '../Model_results/lp_fig_cnn.json')

        pio.write_json(cm_fig_bilstm, '../Model_results/cm_fig_bilstm.json')
        pio.write_json(roc_fig_bilstm, '../Model_results/roc_fig_bilstm.json')
        pio.write_json(lp_fig_bilstm, '../Model_results/lp_fig_bilstm.json')

        client.close()
    finally:
        client.close()
