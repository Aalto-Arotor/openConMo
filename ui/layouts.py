from dash import html, dcc
import dash_mantine_components as dmc

def create_left_panel():
    """Create the left panel with controls and metadata display"""
    return dmc.Group([
        # Left side - Upload and Metadata
        dmc.Stack([
            # Upload section
            dmc.Paper([
                dmc.Title("Upload Data", order=3),
                dcc.Upload(
                    id='upload-data',
                    children=dmc.Group([
                        dmc.Text('Drag and Drop or'),
                        dmc.Text('Select Parquet file', c="blue", fw=500),
                    ], position="center"),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0',
                    },
                    accept='.parquet',
                    multiple=False
                ),
            ], p="md", style={'width': '380px'}),
            
            # Metadata section
            dmc.Paper([
                dmc.Title("Measurement Info", order=3),
                dmc.Paper(
                    id='metadata-display',
                    children=[
                        dmc.Text("No file uploaded yet", c="dimmed")
                    ],
                    p="md",
                    withBorder=True,
                    mt="sm"
                )
            ], p="md", style={'width': '380px'})
        ], spacing="md"),
        
        # Right side - Options Panel
        dmc.Paper([
            dmc.Title("Choose method", order=3),
            dmc.Stack([
                dmc.Select(
                    id='dummy-dropdown-1',
                    data=[
                        {'label': 'Method 1 (envelope)', 'value': '1'},
                        {'label': 'Method 2', 'value': '2'},
                        {'label': 'Method 3', 'value': '3'}
                    ],
                    value='1',
                ),
                
                # Time range inputs
                dmc.Text("Time Range (seconds)", fw=500, mt="sm"),
                dmc.Group([
                    dmc.NumberInput(
                        id='time-start',
                        value=0,
                        step=1,
                        precision=2,
                        label="Start",
                        style={'width': '44%'}
                    ),
                    dmc.NumberInput(
                        id='time-stop',
                        value=10,
                        step=1,
                        precision=2,
                        label="Stop",
                        style={'width': '44%'}
                    ),
                ]),
                
                # Limits selection
                dmc.Text("Spectrum limits", fw=500, mt="sm"),
                dmc.Group([
                    dmc.NumberInput(
                        id='x_lim_1',
                        value=0,
                        step=100,
                        precision=2,
                        label="x min",
                        style={'width': '20%'}
                    ),
                    dmc.NumberInput(
                        id='x_lim_2',
                        value=0,
                        step=100,
                        precision=0,
                        label="x max",
                        style={'width': '20%'}
                    ),
                    dmc.NumberInput(
                        id='y_lim_1',
                        value=0,
                        step=0.01,
                        precision=4,
                        label="y min",
                        style={'width': '20%'}
                    ),
                    dmc.NumberInput(
                        id='y_lim_2',
                        value=0,
                        step=0.01,
                        precision=4,
                        label="y max",
                        style={'width': '20%'}
                    ),
                ]),
                # Cursor selection
                dmc.Text("Cursor selection", fw=500, mt="sm"),
                dmc.Group([
                    dmc.NumberInput(
                        id='ff_hz',
                        value=0,
                        step=0.001,
                        precision=3,
                        label="FF (X rot. freq)",
                        style={'width': '27%'}
                    ),
                    dmc.NumberInput(
                        id='n_harmonics',
                        value=5,
                        step=1,
                        precision=0,
                        label="N harmonics",
                        style={'width': '27%'}
                    ),
                    dmc.NumberInput(
                        id='f_sb_hz',
                        value=0,
                        step=0.01,
                        precision=2,
                        label="SB (Hz)",
                        style={'width': '27%'}
                    ),
                ]),
                
                # Scale controls
                dmc.Text("Frequency Scale", fw=500, mt="sm"),
                dmc.SegmentedControl(
                    id='freq-scale',
                    data=[
                        {'label': 'Linear', 'value': 'linear'},
                        {'label': 'Logarithmic', 'value': 'log'}
                    ],
                    value='linear',
                    fullWidth=True
                ),
                
                dmc.Text("Amplitude Scale", fw=500, mt="sm"),
                dmc.SegmentedControl(
                    id='amp-scale',
                    data=[
                        {'label': 'Linear', 'value': 'linear'},
                        {'label': 'Logarithmic', 'value': 'log'}
                    ],
                    value='linear',
                    fullWidth=True
                ),
            ], spacing="sm"),
        ], p="md", style={'width': '400px'}, withBorder=True)
    ], align="flex-start", spacing="md")

def create_plot_options():
    """Create the plot options section"""
    return html.Div([
        html.Label("Dummy Option 1"),
        dcc.Dropdown(
            id='dummy-dropdown-1',
            options=[
                {'label': 'Option A', 'value': 'A'},
                {'label': 'Option B', 'value': 'B'}
            ],
            value='A'
        ),
        html.Label("Common options:", style={'marginTop': '10px'}),
        create_time_range_inputs(),
        html.Label("Frequency domain options:", style={'marginTop': '10px'}),
        create_scale_controls()
    ])

def create_time_range_inputs():
    """Create the time range input section"""
    return html.Div([
        html.Label("Time Range (seconds)", style={'marginTop': '10px'}),
        html.Div([
            html.Div([
                html.Label("Start", style={'fontSize': '0.9em'}),
                dcc.Input(
                    id='time-start',
                    type='number',
                    value=0,
                    step=0.1,
                    precision=2,
                    style={'width': '100%'}
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Stop", style={'fontSize': '0.9em'}),
                dcc.Input(
                    id='time-stop',
                    type='number',
                    value=10,
                    step=0.1,
                    precision=2,
                    style={'width': '100%'}
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ], style={'display': 'flex', 'alignItems': 'flex-end'})
    ])

def create_scale_controls():
    """Create the scale control radio buttons"""
    return html.Div([
        html.Label("Frequency Scale", style={'marginTop': '15px'}),
        dcc.RadioItems(
            id='freq-scale',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='log',
            inline=True,
            style={'display': 'flex', 'justifyContent': 'space-around', 'width': '100%'}
        ),
        html.Label("Amplitude Scale", style={'marginTop': '15px'}),
        dcc.RadioItems(
            id='amp-scale',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='linear',
            inline=True,
            style={'display': 'flex', 'justifyContent': 'space-around', 'width': '100%'}
        )
    ])

def create_metadata_section():
    """Create the metadata display section"""
    return html.Div([
        html.H4("Measurement Info", style={'marginTop': '20px'}),
        html.Div(id='metadata-display', children=[
            html.P("No file uploaded yet", style={'color': '#666'})
        ], style={
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'marginTop': '10px'
        })
    ]) 