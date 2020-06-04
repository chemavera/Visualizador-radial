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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from calculate_mapping_general import *
from sklearn.preprocessing import LabelEncoder

import json
import datetime
import operator
import os

import base64
import io


def commonFunc(data, algo, graphics_algo, vector_norm,
               vector_choose, selected, r_display,
               axis_data=None, choose_target=None, item_color=None):
    df = pd.DataFrame(data)
    
    def get_index(name):
        '''
        return the index of a column name
        '''
        for column in df.columns:
            # print(f"{selected2} + {df.columns.get_loc(selected2)}")
            if column == name:
                index = df.columns.get_loc(column)
                # print(index)
                return index

    def interact(V, row, new_values):
    	V[row][0] = new_values[0]
    	V[row][1] = new_values[1]
    	# V = np.zeros_like(V)
    	return V
        
    result=[]
    for i  in range(len(selected)):
        X = get_index(selected[i])
        result.append(X)

    dx = df[df.columns[result]]
    x = dx.values
    x = (x).astype(float)
    
    Targ = get_index(choose_target)
    dtarg = df[df.columns[Targ]]
    tg = dtarg.values
    #Codificación del Target
    encoder = LabelEncoder()
    encoder.fit(tg)
    targ_c = encoder.transform(tg)
    
    n_components = len(selected) 
    k = get_index(vector_choose)
    W = np.identity(n_components)
    
    #Estandarizamos los datos y se calcula la matriz traspuesta.
    X_std = stats.zscore(x)
    colors = ['#0D76BF', '#00cc96','#EF553B', '#9D13EB', '#00C2C0', '#56CC00', 
              '#93ACC2', '#B9EBC2', '#F09F9E', '#D5A6EB','#F5EECD', '#F5C097', 
              '#007CAD', '#00BA16', '#C96614', '#C70000', '#4CFF00', '#EBFF01',
              '#FF0010', '#FF7400']

    if algo == "PCA":
        #Aplicamos la reducción de la dimensionalidad.
        pca = PCA()
        pca.fit_transform(X_std)
        V_r = np.transpose(pca.components_[0:2, :])

        # Modify according to interaction data
        if axis_data:
        	for axis in axis_data.keys():
        		V_r = interact(V_r, int(axis), axis_data[axis])
        
        if graphics_algo == "RadViz":
            P = mapping('RadViz', X_std, V_r, W, 1, k)
        elif graphics_algo == "SC":
            P = mapping('SC', X_std, V_r, W, 1, k)
        elif graphics_algo == "Adaptable":
            if vector_norm == 1:
                P = mapping('Adaptable', X_std, V_r, W, 1, k)
            else:
                P = mapping('Adaptable', X_std, V_r, W, 'Inf', k)
        elif graphics_algo == "Adaptable exact":
            if vector_norm == 1:
                P = mapping('Adaptable exact', X_std, V_r, W, 1, k)
            elif vector_norm == 2:
                P = mapping('Adaptable exact', X_std, V_r, W, 2, k)
            else:
                P = mapping('Adaptable exact', X_std, V_r, W, 'Inf', k)
        elif graphics_algo == "Adaptable ordered":
            if vector_norm == 1:
                P = mapping('Adaptable ordered', X_std, V_r, W, 1, k)
            elif vector_norm == 2:
                P = mapping('Adaptable ordered', X_std, V_r, W, 2, k)
            elif vector_norm == 'Inf':
                P = mapping('Adaptable ordered', X_std, V_r, W, 'Inf', k)
    elif algo == "LDA":
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_std,targ_c)
        V_r = lda.scalings_
        
        # Modify according to interaction data
        if axis_data:
            for axis in axis_data.keys():
                V_r = interact(V_r, int(axis), axis_data[axis])
                
        if graphics_algo == "RadViz":
                P = mapping('RadViz', X_std, V_r, W, 1, k)
        elif graphics_algo == "SC":
                P = mapping('SC', X_std, V_r, W, 1, k)
        elif graphics_algo == "Adaptable":
                if vector_norm == 1:
                    P = mapping('Adaptable', X_std, V_r, W, 1, k)
                else:
                    P = mapping('Adaptable', X_std, V_r, W, 'Inf', k)
        elif graphics_algo == "Adaptable exact":
                if vector_norm == 1:
                    P = mapping('Adaptable exact', X_std, V_r, W, 1, k)
                elif vector_norm == 2:
                    P = mapping('Adaptable exact', X_std, V_r, W, 2, k)
                elif vector_norm == 'Inf':
                    P = mapping('Adaptable exact', X_std, V_r, W, 'Inf', k)
        elif graphics_algo == "Adaptable ordered":
                if vector_norm == 1:
                    P = mapping('Adaptable ordered', X_std, V_r, W, 1, k)
                elif vector_norm == 2:
                    P = mapping('Adaptable ordered', X_std, V_r, W, 2, k)
                elif vector_norm == 'Inf':
                    P = mapping('Adaptable ordered', X_std, V_r, W, 'Inf', k)
    
    #Se pasa de una matriz a un array.
    P = np.asarray(P)
    
    #Se calcula R y se ordena según este.
    R = np.sqrt((np.array(P[:,0]))**2 + (np.array(P[:,1]))**2)
    
    r_graph = dict(zip(R, P))

    r_targ = dict(zip(R, tg))

    resultado = sorted(r_graph.items(), key=operator.itemgetter(0))

    resultadot = sorted(r_targ.items(), key=operator.itemgetter(0))
    
    R = sorted(R)

    theta = []
    for i in range(len(V_r)):
        if V_r[i,1] <= 0 and V_r[i,0] <= 0:
            th = math.atan(V_r[i,1]/V_r[i,0])
            th = ((3*math.pi)/2) - th
            theta.append(th)
        elif V_r[i,1] <= 0 and V_r[i,0] >= 0:
            th = math.atan(V_r[i,1]/V_r[i,0])
            th = (2*math.pi) + th
            theta.append(th)
        elif V_r[i,1] >= 0 and V_r[i,0] <= 0:
            th = math.atan(V_r[i,1]/V_r[i,0])
            th = math.pi + th
            theta.append(th)
        else:
            th = math.atan(V_r[i,1]/V_r[i,0])
            theta.append(th)
    
    theta = np.asarray(theta)
    
    #Se construyen los arrays a visualizar.
    
    P_p = []
    targ = []
    for i in range(len(resultado)):
    	# check values are in the allowed range
    	if R[i] >= r_display[0] and R[i] <= r_display[1]:
            p_P = ((resultado[i][1][0]), (resultado[i][1][1]))
            trg = resultadot[i][1]
            P_p.append(p_P)
            targ.append(trg)
    
    P_p = np.asarray(P_p)
    targ = np.asarray(targ)
    
    
    # Se calcula el mayor radio.
    mayor = 0
    for i in range(len(V_r)):
        V_rr = math.sqrt((V_r[i,0])**2 + (V_r[i,1])**2)
        if V_rr > mayor:
            mayor = V_rr
        else:
            mayor

    for i in range(len(R)):
        if R[i] > mayor:
            mayor = R[i]
        else:
            mayor
    
    
    
    # Datos para la trama circular.
    cir = []
    for i in range(0, 201, 1):
        cr = [(i*math.pi)/100]
        cir.append(cr)

    cir = np.array(cir)

    x_cir = []
    y_cir = []
    for i in range(len(cir)):    
        x = (mayor*1.1)*math.cos(cir[i])
        y= (mayor*1.1)*math.sin(cir[i])
        x_cir.append(x)
        y_cir.append(y)
    
    x_cir = np.array(x_cir)
    y_cir = np.array(y_cir)
    
    if df.empty:
        raise dash.exceptions.PreventUpdate()
    else:
        if item_color == 'values':
            pi_2 = math.pi/2
            tres_pi_2 = (3*math.pi)/2
            
            trace = []
            #for i in range(len(P_p)):
            trace1 = go.Scatter(
                    x=P_p[:,0],
                    y=P_p[:,1],
                    mode='markers',
                    hoverinfo='skip',
                    showlegend = False,
                    marker=dict(
                        size=8,
                        cmax=R[-1],
                        cmin=R[0],
                        color=R,
                        opacity=0.8,
                        colorbar=dict(
                            title="Value r:"
                        ),
                        colorscale=[[0, "rgb(4,173,220)"],
                                    [1, "#EF553B"]],
                        )
                )
            trace.append(trace1)
                                    
            
            for i in range(len(theta)):
                if theta[i] == tres_pi_2:
                    pos = 'bottom center'
                elif theta[i] == pi_2:
                    pos = 'top center'
                elif theta[i] > pi_2 and theta[i] < tres_pi_2: 
                    pos = 'middle left'
                else:
                    pos = 'middle right'
                
                trace2 = go.Scatter(
                    x = [(mayor*1.12)*math.cos(theta[i])],
                    y = [(mayor*1.12)*math.sin(theta[i])],
                    hoverinfo='skip',
                    mode="markers + text",
                    marker = dict( size = 5,
                    color = "rgba(245,208,0,0.8)"),
                    showlegend = False,
                    line = dict(
                        color = "rgba(245,208,0,0.8)",
                        width = 4),
                    text = df.columns[i],
                    textposition=pos,
                    textfont=dict(
                        family="helvetica",
                        size=15,
                        color="rgba(245,208,0,0.8)"
                        )
                    )
                trace.append(trace2)
            
            trace3 = go.Scatter(
                x=x_cir,
                y=y_cir,
                fill='tozeroy',
                mode= 'none',
                fillcolor = "rgba(245,208,0,0.1)",
                showlegend=False,
                hoverinfo='skip',
                )
            trace.append(trace3)
            
            trace4 = go.Scatter(
                x=x_cir*1.7,
                y=y_cir*1.7,
                fill='tozeroy',
                mode= 'none',
                fillcolor = "rgba(4,173,220,0.0001)",
                showlegend=False,
                hoverinfo='skip',
                )
            trace.append(trace4)
            
            for i in range(len(V_r)):
                trace5 = go.Scatter(
                    x = [0,V_r[i,0]],
                    y = [0,V_r[i,1]],
                    marker = dict( size = 1,
                    color = "rgba(245,208,0,0.8)"),
                    showlegend = False,
                    line = dict(
                        color = "rgba(245,208,0,0.8)",
                        width = 3),
                    text = df.columns[i],
                    textposition="top right",
                    textfont=dict(
                        family="helvetica",
                    size=15,
                    color="rgba(245,208,0,0.8)"
                    )
                )
                trace.append(trace5)
                
            vectors=[]
            for i in range(len(V_r)):
                layout = {
                        'type': 'line',
                        'x0': 0,
                        'x1': V_r[i,0],

                        'y0': 0,
                        'y1': V_r[i,1],
                        

                        'line': {
                                'width': 3,
                                'color': 'rgba(245,208,0,0.8)'
                                },
                        'height': 225,
                        }
                        
                vectors.append(layout)
            
        else:
            pi_2 = math.pi/2
            tres_pi_2 = (3*math.pi)/2
            
            trace = []
            for name, col in zip(list(set(targ)), colors):
            
                trace1 = go.Scatter(
                    x=P_p[targ==name,0],
                    y=P_p[targ==name,1],
                    mode='markers',
                    name=str(name),
                    hoverinfo='skip',
                    marker=dict(
                        color=col,
                        size=12,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0.5),
                        opacity=0.8)
                    )
                trace.append(trace1)
                                    
            
            for i in range(len(theta)):
                if theta[i] == tres_pi_2:
                    pos = 'bottom center'
                elif theta[i] == pi_2:
                    pos = 'top center'
                elif theta[i] > pi_2 and theta[i] < tres_pi_2: 
                    pos = 'middle left'
                else:
                    pos = 'middle right'
                
                trace2 = go.Scatter(
                    x = [(mayor*1.12)*math.cos(theta[i])],
                    y = [(mayor*1.12)*math.sin(theta[i])],
                    hoverinfo='skip',
                    mode="markers + text",
                    marker = dict( size = 5,
                    color = "rgba(245,208,0,0.8)"),
                    showlegend = False,
                    line = dict(
                        color = "rgba(245,208,0,0.8)",
                        width = 4),
                    text = df.columns[i],
                    textposition=pos,
                    textfont=dict(
                        family="helvetica",
                        size=15,
                        color="rgba(245,208,0,0.8)"
                        )
                    )
                trace.append(trace2)
            
            trace3 = go.Scatter(
                x=x_cir,
                y=y_cir,
                fill='tozeroy',
                mode= 'none',
                fillcolor = "rgba(245,208,0,0.1)",
                showlegend=False,
                hoverinfo='skip',
                )
            trace.append(trace3)
            
            trace4 = go.Scatter(
                x=x_cir*1.7,
                y=y_cir*1.7,
                fill='tozeroy',
                mode= 'none',
                fillcolor = "rgba(4,173,220,0.0001)",
                showlegend=False,
                hoverinfo='skip',
                )
            trace.append(trace4)
            
            for i in range(len(V_r)):
                trace5 = go.Scatter(
                    x = [0,V_r[i,0]],
                    y = [0,V_r[i,1]],
                    marker = dict( size = 1,
                    color = "rgba(245,208,0,0.8)"),
                    showlegend = False,
                    line = dict(
                        color = "rgba(245,208,0,0.8)",
                        width = 3),
                    text = df.columns[i],
                    textposition="top right",
                    textfont=dict(
                        family="helvetica",
                    size=15,
                    color="rgba(245,208,0,0.8)"
                    )
                )
                trace.append(trace5)
                
            vectors=[]
            for i in range(len(V_r)):
                layout = {
                        'type': 'line',
                        'x0': 0,
                        'x1': V_r[i,0],

                        'y0': 0,
                        'y1': V_r[i,1],
                        

                        'line': {
                                'width': 3,
                                'color': 'rgba(245,208,0,0.8)'
                                },
                        'height': 225,
                        }
                        
                vectors.append(layout)
                
        max_r = max(R)
        return {
                'data': trace,
                'layout': {'shapes': vectors,
                           'width' : 800,
                           'height' : 800,
                           'plot_bgcolor': 'rgb(20,24,22)',
                           'paper_bgcolor': 'rgb(20,24,22)',
                           'font': {
                                   'color': 'rgb(4,173,220)'
                                   }}
                }, max_r