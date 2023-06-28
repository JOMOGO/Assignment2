import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
from plotly.graph_objs import Box
import dash_bootstrap_components as dbc
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import joblib
from Database.db_connector import load_data_from_mongodb
import plotly.io as pio

df, collection= load_data_from_mongodb()

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

# Load plots
cm_fig_rnn = pio.read_json('../Model_results/cm_fig_rnn.json')
roc_fig_rnn = pio.read_json('../Model_results/roc_fig_rnn.json')
lp_fig_rnn = pio.read_json('../Model_results/lp_fig_rnn.json')

cm_fig_cnn = pio.read_json('../Model_results/cm_fig_cnn.json')
roc_fig_cnn = pio.read_json('../Model_results/roc_fig_cnn.json')
lp_fig_cnn = pio.read_json('../Model_results/lp_fig_cnn.json')

cm_fig_bilstm = pio.read_json('../Model_results/cm_fig_bilstm.json')
roc_fig_bilstm = pio.read_json('../Model_results/roc_fig_bilstm.json')
lp_fig_bilstm = pio.read_json('../Model_results/lp_fig_bilstm.json')

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
                        html.Br(),
                        dcc.Input(id='user-input', type='text', placeholder='Type a review...'),
                        html.Button('Predict', id='predict-button', n_clicks=0),
                        html.Div(id='prediction-result'),
                        html.Br(),  # To add space
                        html.Label("Model Predictions:"),
                        dash_table.DataTable(
                            id='table',
                            columns=[
                                {"name": "Model", "id": "model"},
                                {"name": "Prediction", "id": "prediction"},
                                {"name": "Certainty (%)", "id": "certainty"}
                            ],
                            data=[],
                            style_cell={
                                'textAlign': 'left',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                        )
                    ])
                ])
            ])
        ]),
        dcc.Tab(label='Model Performance', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.RadioItems(
                            id='model-select',
                            options=[
                                {'label': 'RNN', 'value': 'rnn'},
                                {'label': 'CNN', 'value': 'cnn'},
                                {'label': 'Bi-LSTM', 'value': 'bilstm'}
                            ],
                            value='rnn'
                        ),
                        dcc.Graph(id='cm', config={'displayModeBar': False}),
                        dcc.Graph(id='roc', config={'displayModeBar': False}),
                        dcc.Graph(id='lp', config={'displayModeBar': False})
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
    [Output('prediction-result', 'children'),
     Output('table', 'data')],
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

        # Compute prediction certainty (maximum probability)
        rnn_certainty = np.max(rnn_pred) * 100
        cnn_certainty = np.max(cnn_pred) * 100
        bilstm_certainty = np.max(bilstm_pred) * 100

        # Create response
        response = f'RNN Model Prediction: {rnn_sentiment}, CNN Model Prediction: {cnn_sentiment}, Bi-LSTM Model Prediction: {bilstm_sentiment}'

        # Create table data
        table_data = [
            {"model": "RNN", "prediction": rnn_sentiment, "certainty": f"{rnn_certainty:.2f}%"},
            {"model": "CNN", "prediction": cnn_sentiment, "certainty": f"{cnn_certainty:.2f}%"},
            {"model": "Bi-LSTM", "prediction": bilstm_sentiment, "certainty": f"{bilstm_certainty:.2f}%"},
        ]

        # Compute the overall sentiment (mode of all predictions)
        overall_sentiment = max(set([rnn_sentiment, cnn_sentiment, bilstm_sentiment]), key=[rnn_sentiment, cnn_sentiment, bilstm_sentiment].count)
        overall_certainty = (rnn_certainty + cnn_certainty + bilstm_certainty) / 3
        table_data.append({"model": "Overall", "prediction": overall_sentiment, "certainty": f"{overall_certainty:.2f}%"})

        return response, table_data
    else:
        return dash.no_update, dash.no_update

@app.callback(
    [Output('cm', 'figure'),
     Output('roc', 'figure'),
     Output('lp', 'figure')],
    [Input('model-select', 'value')]
)
def update_performance_plots(selected_model):
    if selected_model == 'rnn':
        return cm_fig_rnn, roc_fig_rnn, lp_fig_rnn
    elif selected_model == 'cnn':
        return cm_fig_cnn, roc_fig_cnn, lp_fig_cnn
    elif selected_model == 'bilstm':
        return cm_fig_bilstm, roc_fig_bilstm, lp_fig_bilstm

if __name__ == '__main__':
    app.run_server(debug=True)
