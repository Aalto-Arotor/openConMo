import dash
from dash import html, dcc
import dash_mantine_components as dmc
from pathlib import Path
from flask import send_from_directory
from layouts import create_layout
from callbacks import register_callbacks

dmc.pre_render_color_scheme()

app = dash.Dash(__name__)

_logo_dir = Path(__file__).resolve().parents[1] / "docs" / "_images"


@app.server.route("/static/openconmo_logo.png")
def serve_openconmo_logo():
    return send_from_directory(_logo_dir, "openconmo_logo.png")

app.layout = dmc.MantineProvider(
    children=[
        dmc.NotificationContainer(id="notifications-container"),
        
        # This calls the AppShell structure defined in your updated layouts.py
        create_layout()
    ],
    # You can set default theme settings here
    theme={
        "primaryColor": "blue",
        "fontFamily": "'Inter', sans-serif",
    }
)

# Register callbacks
register_callbacks(app)

def main():
    # Note: Use dash.run instead of run_server in newer Dash versions (optional but recommended)
    app.run(debug=True)

if __name__ == '__main__':
    main()