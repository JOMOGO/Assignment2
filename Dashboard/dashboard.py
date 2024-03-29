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
import re

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

# Load tokenizer
tokenizer = joblib.load('../Model_results/tokenizer.pkl')

# Load plots
cm_fig_rnn = pio.read_json('../Model_results/cm_fig_rnn.json')
roc_fig_rnn = pio.read_json('../Model_results/roc_fig_rnn.json')

cm_fig_cnn = pio.read_json('../Model_results/cm_fig_cnn.json')
roc_fig_cnn = pio.read_json('../Model_results/roc_fig_cnn.json')

cm_fig_bilstm = pio.read_json('../Model_results/cm_fig_bilstm.json')
roc_fig_bilstm = pio.read_json('../Model_results/roc_fig_bilstm.json')


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
                        html.Br(),
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
                        html.Br(),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("RNN", id="btn-rnn", color="success", className="mr-1"),
                                dbc.Button("CNN", id="btn-cnn", color="primary", className="mr-1"),
                                dbc.Button("Bi-LSTM", id="btn-bilstm", color="primary", className="mr-1"),
                            ],
                            className="mb-3",
                        ),
                        dcc.Graph(id='cm', config={'displayModeBar': False}),
                        dcc.Graph(id='roc', config={'displayModeBar': False}),
                    ])
                ])
            ])
        ])
    ])
])

@app.callback(
    [Output('box-plot', 'figure'),
     Output('bar-chart', 'figure')],
    Input('world-map', 'clickData'))
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
                boxpoints=False
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
    [State('user-input', 'value')])
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Preprocess user input
        processed_input = re.sub(
            r'\b(not|no|never|neither|nothing|none|no one|nobody|nowhere|nor|barely|hardly|scarcely|seldom|rarely)\s+(\w+)\b',r'\1_\2', value)
        # Tokenize and pad preprocessed user input
        sequences = tokenizer.texts_to_sequences([processed_input])
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
     Output('btn-rnn', 'color'),
     Output('btn-cnn', 'color'),
     Output('btn-bilstm', 'color')],
    [Input('btn-rnn', 'n_clicks'),
     Input('btn-cnn', 'n_clicks'),
     Input('btn-bilstm', 'n_clicks')])
def update_performance_plots(n_clicks_rnn, n_clicks_cnn, n_clicks_bilstm):
    ctx = dash.callback_context

    if not ctx.triggered:
        return cm_fig_rnn, roc_fig_rnn, 'success', 'primary', 'primary'  # RNN selected by default
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-rnn':
        return cm_fig_rnn, roc_fig_rnn, 'success', 'primary', 'primary'
    elif button_id == 'btn-cnn':
        return cm_fig_cnn, roc_fig_cnn, 'primary', 'success', 'primary'
    elif button_id == 'btn-bilstm':
        return cm_fig_bilstm, roc_fig_bilstm, 'primary', 'primary', 'success'


if __name__ == '__main__':
    app.run_server(debug=True)
