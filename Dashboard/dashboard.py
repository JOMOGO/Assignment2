from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Dashboard'),
])

if __name__ == '__main__':
    app.run_server(debug=True)