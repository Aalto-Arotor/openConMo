import dash
from dash import html, dcc
import dash_mantine_components as dmc
from layouts import create_left_panel
from callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    dmc.Container([
        dmc.Group([
            # Left panel
            create_left_panel(),
            
            # Right panel for plots
            dmc.Paper([
                dmc.Title("Signal Analysis Plots", order=3, align="Left"),  # Adding a title
                dcc.Graph(id='time-plot'),
                dcc.Graph(id='frequency-plot'),
                dcc.Graph(id='envelope-plot'),
            ], style={'flex': '1'}, p="md", shadow="sm")
        ], grow=False, align="flex-start")
    ], fluid=True, p=0)
])

# Register callbacks
register_callbacks(app)

def main():
    app.run_server(debug=True)

if __name__ == '__main__':
    main()