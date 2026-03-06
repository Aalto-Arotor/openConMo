from pathlib import Path

import dash
import dash_mantine_components as dmc
from callbacks import register_callbacks
from flask import send_from_directory
from layouts import create_layout

dmc.pre_render_color_scheme()

app = dash.Dash(__name__, suppress_callback_exceptions=True)

_logo_dir = Path(__file__).resolve().parents[1] / "docs" / "_images"


@app.server.route("/static/openconmo_logo.png")
def serve_openconmo_logo():
    return send_from_directory(_logo_dir, "openconmo_logo.png")


app.layout = dmc.MantineProvider(
    children=[
        dmc.NotificationContainer(id="notifications-container"),
        create_layout(),
    ],
    theme={
        "primaryColor": "blue",
        "fontFamily": "'Inter', sans-serif",
    },
)

register_callbacks(app)


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
