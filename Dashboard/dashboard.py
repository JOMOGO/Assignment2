import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output
from pymongo import MongoClient
import plotly.express as px
from plotly.graph_objs import Box
import dash_bootstrap_components as dbc
from Database.db_connector import load_data_from_mongodb

df = load_data_from_mongodb()

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Prepare data for the world map
map_data = df['Reviewer_Nationality'].value_counts().reset_index()
map_data.columns = ['Country', 'Count']

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


if __name__ == '__main__':
    app.run_server(debug=True)
