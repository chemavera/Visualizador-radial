# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:07:42 2019

@author: JoséManuel
"""

import dash_html_components as html
import dash_core_components as dcc
import dash

import plotly
import plotly.graph_objects as go
import dash_table as dte
import dash_table
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

import numpy as np
import numpy.matlib
import math
from scipy import stats
from sklearn.decomposition import PCA
from calculate_mapping_general import *
from common_func import *

import json
import datetime
import operator
import os

import base64
import io


app = dash.Dash(__name__)

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

all_options = {
        'SC':[],
        'RadViz':[],
        'Adaptable': [1, 2,'Inf'],
        'Adaptable exact': [1, 2, 'Inf'],
        'Adaptable ordered': [1, 2, 'Inf']
    
}

app.layout = html.Div(className="app-header--title", children = [

    html.Div(id="left",
        children = [html.Div([dcc.Graph(id='datatable-upload-graph', config={
                            'scrollZoom': True,
                            'editable': True,
                            'edits': {
                                    'shapePosition': True
                                    },
                            }),#"autosizable": True}),
                    ], style={'width': '100%'}),
    

        html.H4("", id='relayout-data'),

        dcc.RangeSlider(
                    id='my-range-slider',
                    min=0,
                    max=20,
                    marks={i: 'R:{}'.format(i) for i in range(0, int(20), 1)},
                    step=(20)/100,
                    value=[0, 100]
                 )]
    ),
    
    html.Div(id='right',
    
            children= [
        
            #html.H4("Upload Files"),
            html.Button(dcc.Upload(
                id='upload-data',
                children='Drag and Drop or Select Files',
                multiple=False),
            ),

            html.Br(),
            html.Button(
            id='propagate-button',
            n_clicks=0,
            children='Propagate Table Data',
            #color="primary"
            ),


            html.Br(),
            html.H4("Filter Column"),
            dcc.Dropdown(id='dropdown_table_filterColumn',
                multi = True,
                placeholder='Filter Column'),

            html.H4("Initialization Algorithm"),
            dcc.Dropdown(
                id='algo-selector',
                options=[
                    {'label': 'PCA', 'value': 'PCA'},
                    {'label': 'LDA', 'value': 'LDA'}
                ],
                value='PCA'
            ),

            html.H4("Radial Visualization"),
            
            dcc.RadioItems(
                id='graphics-algo-selector',
                options=[{'label': k, 'value': k} for k in all_options.keys()],
                value='SC'
            ),

            dcc.RadioItems(id='vector-norm'),
            
            dcc.Dropdown(
                id='vector-choose',
                placeholder='Choose a vector',
                multi = False,
            ),
                    
            html.H4("Select target"),
                    
            dcc.Dropdown(
                id='choose-target',
                placeholder='choose your target',
                multi = False,
            ),
                    
            html.H4("Select color item"),
            
            dcc.RadioItems(
                    id='item-color',
                    options=[
                            {'label': 'Target', 'value': 'target'},
                            {'label': 'Values', 'value': 'values'}
                            ],
                    value='target'
            ) ,

            html.Br(),
            html.Button(
                id='calculate-button',
                n_clicks=0,
                children='Calculate',
                className="button"
            ),

            html.Div(dte.DataTable(data=[{}], id='table'))
    ])
    
    
])


# Functions

# file upload function
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df



# callback table creation
@app.callback(Output('table', 'data'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            return df.to_dict('records')
        else:
            return [{}]
    else:
        return [{}]


#callback update options of filter dropdown
@app.callback(Output('dropdown_table_filterColumn', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows) # <- problem! dff stays empty even though table was uploaded

        print ("updating... dff empty?:"), dff.empty #result is True, labels stay empty

        return [{'label': i, 'value': i} for i in (list(dff))]

@app.callback(
    dash.dependencies.Output('vector-norm', 'options'),
    [dash.dependencies.Input('graphics-algo-selector', 'value')])
def set_graphics_selector(selected):
    return [{'label': i, 'value': i} for i in all_options[selected]]


@app.callback(
    dash.dependencies.Output('vector-norm', 'value'),
    [dash.dependencies.Input('vector-norm', 'options')])
def set_vector_norm(available_options):
    return available_options[0]['value']

@app.callback(Output('vector-choose', 'options'),
              [Input('dropdown_table_filterColumn', 'value')])
def choose_vector_options(vector_selected):
    if vector_selected == [{}]:
        pass
    else:
        return [{'label': i, 'value': i} for i in vector_selected] 

@app.callback(Output('vector-choose', 'value'),
              [Input('vector-choose', 'options')])
def choose_vector_value(available_options):
    if available_options == [{}]:
        pass
    else:
        return available_options[0]['value']
     
@app.callback(Output('choose-target', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('table', 'data')])
def choose_target(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows) # <- problem! dff stays empty even though table was uploaded

        print ("updating... dff empty?:"), dff.empty #result is True, labels stay empty

        return [{'label': i, 'value': i} for i in (list(dff))]

@app.callback(Output('datatable-upload-graph', 'figure'),
              [Input('calculate-button', 'n_clicks'),
               Input('relayout-data', 'value'),
               Input('my-range-slider', 'value')],
              [State('table', 'data'),
               State('algo-selector', 'value'),
               State('graphics-algo-selector', 'value'),
               State('vector-norm', 'value'),
               State('vector-choose', 'value'),
               State('dropdown_table_filterColumn','value'),
               State('choose-target', 'value'),
               State('item-color', 'value')])
def display_graph(n_clicks, axis_data, r_display, dated, algo, graphics_algo, vector_norm, vector_choose, selected, choose_target, item_color):
    ''' 
    update the graph from the dopdown selected columns
    '''

    if n_clicks < 1:
        print("Didn't calculate anything")
        return {
                'data': [
                    go.Scatter(
                            x=[0],
                            y=[0],
                            mode='markers',
                        )
                    ],
                'layout': {
                        'width' : 800,
                        'height' : 800,
                        'plot_bgcolor': 'rgb(20,24,22)',
                        'paper_bgcolor': 'rgb(20,24,22)',
                        'font': {
                                'color': 'rgb(4,173,220)'
                                }
                        }
                }

    if axis_data:
        dataprep = [d if d != "'" else '"' for d in axis_data]
        axis_data = ''.join(dataprep)
        axis_data     = json.loads(axis_data)
        
        # idea to update range dinamically?
    
    graph_data, r = commonFunc(dated, algo, graphics_algo,
                      vector_norm, vector_choose, selected, r_display, axis_data, choose_target, item_color)

    return graph_data

@app.callback(
    Output('relayout-data', 'value'),
    [Input('datatable-upload-graph', 'relayoutData')],
    [State('relayout-data', 'value')])
def display_selected_data(relayoutData, own_data):
    # return relayoutData
    data     = json.loads(json.dumps(relayoutData, indent=2))
    if own_data:
        if "autosize" in own_data:
            own_data = {}
        else:
            own_dated = [d if d != "'" else '"' for d in own_data]
            own_data = ''.join(own_dated)
            own_data     = json.loads(own_data)
    else:
        own_data = {}

    # check que hay una redimension de componente
    shapes = False
    for key in data.keys():
        if "shapes[" in key:
            shapes = True
            break
    # Solo continua si se cumple todo bien:
    if shapes == True:
        row = 0
        coords = [0, 0]
        for key in data.keys():
            if "x" in key:
                coords[0] = data[key]
                row = int(key.split("[")[1].split("]")[0])
            elif "y" in key:
                coords[1] = data[key]

        x = coords[0]
        y = coords[1]
        

        data = {str(row): [x, y]}
    
        own_data.update(data) 

    return str(own_data)


app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

if __name__ == '__main__':
    app.run_server(debug = True, port=5000)
