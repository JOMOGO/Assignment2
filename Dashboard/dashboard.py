import dash
from dash import dcc, html, Input, Output, State
from pymongo import MongoClient
import plotly.express as px
from plotly.graph_objs import Box
import dash_bootstrap_components as dbc
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import joblib
from Database.db_connector import load_data_from_mongodb

df = load_data_from_mongodb()

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Prepare data for the world map
map_data = df['Reviewer_Nationality'].value_counts().reset_index()
map_data.columns = ['Country', 'Count']

# Set a maximum length for input text. This should be the same as the maxlen used while training the models.
MAX_LEN = 300

# Load models (replace with paths to your models)
rnn_model = load_model('../Model_results/rnn_model.h5')
cnn_model = load_model('../Model_results/cnn_model.h5')
bilstm_model = load_model('../Model_results/bilstm_model.h5')

# Load tokenizer and vectorizer
tokenizer = joblib.load('../Model_results/tokenizer.pkl')
vectorizer = joblib.load('../Model_results/vectorizer.pkl')

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Score by Country', children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='world-map',
                        figure=px.choropleth(
                            map_data,
                            locations='Country',
                            locationmode='country names',
                            color='Count',
                            title='Number of Reviews by Country',
                            hover_name='Country',
                            labels={'Count': 'Number of Reviews'}
                        ),
                        style={'height': '50vh'}
                    )
                ], width=12),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='box-plot')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='bar-chart')
                    ], width=6)
                ])
            ])
        ]),
        dcc.Tab(label='Sentiment Prediction', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Input(id='user-input', type='text', placeholder='Type a review...'),
                        html.Button('Predict', id='predict-button', n_clicks=0),
                        html.Div(id='prediction-result')
                    ])
                ])
            ])
        ])
    ])
])

@app.callback(
    [Output('box-plot', 'figure'),
     Output('bar-chart', 'figure')],
    Input('world-map', 'clickData')
)
def update_plots(clickData):
    if clickData is None:
        nationality = df['Reviewer_Nationality'].mode()[0]
    else:
        nationality = clickData['points'][0]['location']

    filtered_df = df[df['Reviewer_Nationality'] == nationality].copy()

    # Convert ObjectId to string
    filtered_df['_id'] = filtered_df['_id'].astype(str)

    # Create box plot
    box_plot = {
        'data': [
            Box(
                y=filtered_df["Reviewer_Score"],
                name=f'Reviewer Score Distribution for {nationality}',
                boxpoints=False  # do not show individual points
            )
        ]
    }

    # Create horizontal bar chart of individual review scores
    bar_chart = px.histogram(filtered_df, y="Reviewer_Score", nbins=50, orientation='h',
                             title=f'Reviewer Score Distribution for {nationality}')

    return box_plot, bar_chart

@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Tokenize and pad user input
        sequences = tokenizer.texts_to_sequences([value])
        data = pad_sequences(sequences, maxlen=MAX_LEN)

        # Make predictions
        rnn_pred = rnn_model.predict(data)[0]
        cnn_pred = cnn_model.predict(data)[0]
        bilstm_pred = bilstm_model.predict(data)[0]

        print('Raw predictions:', rnn_pred, cnn_pred, bilstm_pred)

        # Convert predictions to positive/negative (assuming 0 is negative, 1 is positive)
        rnn_sentiment = 'positive' if np.argmax(rnn_pred) else 'negative'
        cnn_sentiment = 'positive' if np.argmax(cnn_pred) else 'negative'
        bilstm_sentiment = 'positive' if np.argmax(bilstm_pred) else 'negative'

        # Create response
        response = f'RNN Model Prediction: {rnn_sentiment}, CNN Model Prediction: {cnn_sentiment}, Bi-LSTM Model Prediction: {bilstm_sentiment}'

        return response

negative_reviews = ["The service was terrible!", "I hated the food, it was so bad!", "This was the worst hotel I've ever stayed at!"]

for review in negative_reviews:
    sequences = tokenizer.texts_to_sequences([review])
    data = pad_sequences(sequences, maxlen=MAX_LEN)
    rnn_pred = rnn_model.predict(data)[0]
    cnn_pred = cnn_model.predict(data)[0]
    bilstm_pred = bilstm_model.predict(data)[0]

    rnn_sentiment = 'positive' if np.argmax(rnn_pred) else 'negative'
    cnn_sentiment = 'positive' if np.argmax(cnn_pred) else 'negative'
    bilstm_sentiment = 'positive' if np.argmax(bilstm_pred) else 'negative'

    print(f"Review: {review}")
    print(f"RNN Prediction: {rnn_sentiment}, CNN Prediction: {cnn_sentiment}, Bi-LSTM Prediction: {bilstm_sentiment}")
    
if __name__ == '__main__':
    app.run_server(debug=True)
