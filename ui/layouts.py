import dash_mantine_components as dmc
from dash import dcc, html


def create_top_panel():
    """Restored 3-column layout with all original
    inputs and DMC v2.6 syntax."""
    return dmc.Group(
        align="flex-start",
        gap="md",
        children=[
            # COLUMN 1: Bearing Fault Frequency Calculator
            dmc.Paper(
                [
                    dmc.Title("Fault Frequency Calculator", order=3, mb="sm"),
                    dmc.Stack(
                        [
                            dmc.NumberInput(
                                id="bearing-speed-rpm",
                                label="Shaft speed (RPM)",
                                value=0,
                                min=0,
                                step=10,
                                w=260,
                            ),
                            dmc.NumberInput(
                                id="bearing-n-rollers",
                                label="Number of rolling elements (N)",
                                value=0,
                                min=1,
                                step=1,
                                decimalScale=0,
                                w=260,
                            ),
                            dmc.NumberInput(
                                id="bearing-ball-d-mm",
                                label="Rolling element diameter d (mm)",
                                value=0,
                                min=0,
                                step=0.1,
                                decimalScale=3,
                                w=260,
                            ),
                            dmc.NumberInput(
                                id="bearing-pitch-d-mm",
                                label="Pitch diameter D (mm)",
                                value=0,
                                min=0,
                                step=0.1,
                                decimalScale=3,
                                w=260,
                            ),
                            dmc.NumberInput(
                                id="bearing-contact-angle-deg",
                                label="Contact angle θ (deg)",
                                value=0.0,
                                step=0.1,
                                decimalScale=3,
                                w=260,
                            ),
                            dmc.Button(
                                "Calculate",
                                id="bearing-calc-btn",
                                n_clicks=0,
                                w=260,
                            ),
                        ],
                        gap="xs",
                    ),
                    dmc.Divider(my="xs"),
                    dmc.Paper(
                        id="bearing-fault-results",
                        p="sm",
                        withBorder=True,
                        children=dmc.Text(
                            "No calculation yet.", c="dimmed", size="xs"
                        ),
                    ),
                    dcc.Store(id="bearing-freq-store", data=None),
                ],
                p="sm",
                withBorder=True,
                radius="sm",
            ),
            # COLUMN 2: Upload and Metadata
            dmc.Stack(
                [
                    dmc.Paper(
                        [
                            dmc.Title("Upload Data", order=3),
                            dcc.Upload(
                                id="upload-data",
                                children=dmc.Group(
                                    [
                                        dmc.Text("Drag and Drop or"),
                                        dmc.Text(
                                            "Select Parquet file",
                                            c="blue",
                                            fw=500,
                                        ),
                                    ],
                                    justify="center",
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px 0",
                                },
                                accept=".parquet",
                                multiple=False,
                            ),
                        ],
                        p="sm",
                        w=380,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("Measurement Info", order=3),
                            dmc.Paper(
                                id="metadata-display",
                                children=[
                                    dmc.Text(
                                        "No file uploaded yet", c="dimmed"
                                    )
                                ],
                                p="sm",
                                withBorder=True,
                                mt="sm",
                                mih=180,
                            ),
                            dcc.Store(id="parquet-freq-store", data=None),
                        ],
                        p="sm",
                        w=380,
                    ),
                ],
                gap="sm",
            ),
            # COLUMN 3: Analysis Method and Signal Options
            dmc.Paper(
                [
                    dmc.Title("Method", order=3, mb="sm"),
                    dmc.Stack(
                        [
                            dmc.Select(
                                id="dummy-dropdown-1",
                                data=[
                                    {
                                        "label": "Envelope",
                                        "value": "1",
                                    },
                                    {
                                        "label": "Cepstrum prewhitening",
                                        "value": "2",
                                    },
                                    {
                                        "label": "Benchmark",
                                        "value": "3",
                                    },
                                ],
                                value="1",
                                clearable=False,
                                allowDeselect=False,
                            ),
                            dmc.Text("Time Range (seconds)", fw=500, mt="xs"),
                            dmc.Group(
                                [
                                    dmc.NumberInput(
                                        id="time-start",
                                        value=1,
                                        step=1,
                                        decimalScale=2,
                                        label="Start",
                                        w="47%",
                                    ),
                                    dmc.NumberInput(
                                        id="time-stop",
                                        value=2,
                                        step=1,
                                        decimalScale=2,
                                        label="Stop",
                                        w="47%",
                                    ),
                                ],
                                justify="apart",
                            ),
                            dmc.Text("Spectrum limits", fw=500, mt="xs"),
                            dmc.Group(
                                [
                                    dmc.NumberInput(
                                        id="x_lim_1",
                                        value=0,
                                        step=100,
                                        decimalScale=2,
                                        label="x min",
                                        w="22%",
                                    ),
                                    dmc.NumberInput(
                                        id="x_lim_2",
                                        value=500,
                                        step=100,
                                        decimalScale=0,
                                        label="x max",
                                        w="22%",
                                    ),
                                    dmc.NumberInput(
                                        id="y_lim_1",
                                        value=0,
                                        step=0.01,
                                        decimalScale=4,
                                        label="y min",
                                        w="22%",
                                    ),
                                    dmc.NumberInput(
                                        id="y_lim_2",
                                        value=0,
                                        step=0.01,
                                        decimalScale=4,
                                        label="y max",
                                        w="22%",
                                    ),
                                ],
                                gap="xs",
                            ),
                            dmc.Text("Cursor selection", fw=500, mt="xs"),
                            dmc.Group(
                                [
                                    dmc.NumberInput(
                                        id="ff_hz",
                                        value=0,
                                        step=0.001,
                                        decimalScale=3,
                                        label="FF (Hz)",
                                        w="23%",
                                    ),
                                    dmc.NumberInput(
                                        id="n_harmonics",
                                        value=3,
                                        step=1,
                                        decimalScale=0,
                                        label="N harm.",
                                        w="23%",
                                    ),
                                    dmc.NumberInput(
                                        id="f_sb_hz",
                                        value=0,
                                        step=0.01,
                                        decimalScale=2,
                                        label="SB (Hz)",
                                        w="23%",
                                    ),
                                    dmc.NumberInput(
                                        id="t_cursor_s",
                                        value=0.5,
                                        step=0.01,
                                        decimalScale=4,
                                        label="T cursor (s)",
                                        w="23%",
                                    ),
                                ],
                                gap="xs",
                            ),
                            dmc.Group(
                                [
                                    dmc.Stack(
                                        [
                                            dmc.Text(
                                                "Frequency Scale",
                                                fw=500,
                                                size="xs",
                                            ),
                                            dmc.SegmentedControl(
                                                id="freq-scale",
                                                data=["linear", "log"],
                                                value="linear",
                                                fullWidth=True,
                                            ),
                                        ],
                                        gap=2,
                                        flex=1,
                                    ),
                                    dmc.Stack(
                                        [
                                            dmc.Text(
                                                "Amplitude Scale",
                                                fw=500,
                                                size="xs",
                                            ),
                                            dmc.SegmentedControl(
                                                id="amp-scale",
                                                data=["linear", "log"],
                                                value="linear",
                                                fullWidth=True,
                                            ),
                                        ],
                                        gap=2,
                                        flex=1,
                                    ),
                                ],
                                gap="md",
                                mt="xs",
                            ),
                            dcc.Store(id="wheel-steps-store", data=None),
                            dcc.Store(id="current-ff-store", data=None),
                            dcc.Store(id="spectrum-x-store", data=None),
                            html.Div(
                                id="spectrum-x-hidden",
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="time-x-hidden", style={"display": "none"}
                            ),
                        ],
                        gap="xs",
                    ),
                ],
                p="sm",
                w=420,
                withBorder=True,
            ),
        ],
    )


def create_header():
    return dmc.AppShellHeader(
        # The style goes HERE to affect the whole top bar
        style={
            "backgroundColor": "rgba(255, 255, 255, 0.3)",
            "backdropFilter": "blur(3px)",
            "WebkitBackdropFilter": "blur(10px)",
            "borderBottom": "1px solid rgba(0, 0, 0, 0.05)",
        },
        children=[
            dmc.Group(
                h="100%",
                px="sm",
                justify="center",
                align="center",
                children=[
                    html.Img(
                        src="/static/openconmo_logo.png",
                        style={
                            "height": "70px",  # Keep height here
                            "width": "auto",
                            "display": "block",
                        },
                    ),
                ],
            )
        ],
    )


def create_footer():
    return dmc.AppShellFooter(
        dmc.Group(
            justify="space-between",
            align="center",
            h="100%",
            px="sm",
            children=[
                dmc.Text("© 2026 Arotor", size="sm", c="dimmed"),
                dmc.Text("v2.6", size="sm", c="dimmed"),
            ],
        )
    )


def create_main_content():
    # Define a common width for both sections
    common_style = {"width": "100%", "maxWidth": "1200px"}

    return dmc.AppShellMain(
        dmc.Container(
            fluid=True,
            children=dmc.Stack(
                align="center",  # Centers both items in the stack
                gap="lg",
                children=[
                    # --- CENTERED CONTROL PANEL ---
                    dmc.Paper(
                        withBorder=True,
                        p="md",
                        radius="md",
                        shadow="xs",
                        style=common_style,  # Using the shared width logic
                        children=[
                            dmc.Group(
                                align="flex-start",
                                gap="xl",
                                wrap="nowrap",
                                children=[
                                    create_top_panel(),
                                ],
                            ),
                        ],
                    ),
                    # --- PLOTS SECTION (Now matches the width) ---
                    dmc.Paper(
                        p="md",
                        withBorder=True,
                        radius="md",
                        shadow="xs",
                        style=common_style,  # Using shared width logic
                        children=[
                            dmc.Title(
                                "Signal Analysis Plots", order=4, mb="md"
                            ),
                            dmc.Stack(
                                [
                                    dcc.Graph(id="envelope-plot"),
                                    dcc.Graph(id="time-plot"),
                                ],
                                gap="md",
                            ),
                        ],
                    ),
                    # --- ROW 3: ABOUT US (Simplified) ---
                    dmc.Paper(
                        p="xl",
                        style={
                            **common_style,
                            "backgroundColor": "transparent",
                        },
                        children=[
                            dmc.Divider(
                                label="About OpenConMo",
                                labelPosition="center",
                                mb="md",
                            ),
                            dmc.Text(
                                (
                                    "OpenConMo is an open-source Python library "  # noqa: E501
                                    "and platform for vibration signal-based "
                                    "condition monitoring developed at Aalto "
                                    "University Rotor Laboratory ARotor."
                                ),
                                size="sm",
                                c="dimmed",
                                ta="center",
                            ),
                            dmc.Group(
                                justify="center",
                                gap="xl",
                                mt="sm",
                                children=[
                                    dmc.Anchor(
                                        "GitHub",
                                        href=(
                                            "https://github.com/"
                                            "Aalto-Arotor/openConMo"
                                        ),
                                        size="xs",
                                        underline=False,
                                    ),
                                    dmc.Anchor(
                                        "Documentation",
                                        href=(
                                            "https://aalto-arotor.github.io/"
                                            "openConMo/"
                                        ),
                                        size="xs",
                                        underline=False,
                                    ),
                                    dmc.Anchor(
                                        "Contact",
                                        href="mailto:arotor.software@aalto.fi",
                                        size="xs",
                                        underline=False,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    )


def create_layout():
    return dmc.AppShell(
        padding="md",
        # 1. Increase height to 100 to fit your 80px-90px logo comfortably
        header={"height": 100},
        footer={"height": 44},
        style={
            "backgroundColor": "#F5F5F5",
        },
        children=[
            create_header(),
            create_footer(),
            create_main_content(),
        ],
    )
