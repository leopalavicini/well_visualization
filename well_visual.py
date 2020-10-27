#!/usr/bin/env python
# coding: utf-8

# # Well Visualizer - FORCE Dataset
#
# As you may already know, FORCE and XEEK released a well log dataset with more than 100 wells for their 2020 Machine Learning contest, with each well contanaing a set of well logs, a facies interpretation and their location.
#
# In this notebook, we are going to create an interactive dashboard for well visualizacion, this incluiding a log view, a map view and a cross-plot. In order to do this, we are going to use Dash and Plotly.
#
# To download the dataset, you can go [here](https://xeek.ai/challenges/force-well-logs/data). The well log data is licensed by [Norwegian License for Open Government Data (NLOD) 2.0.](https://data.norge.no/nlod/en/2.0/) and the facies interpretation done by FORCE is licensed as [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

# ## Data import
# Now, let's import the python packages and the dataset.

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import utm
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# In[2]:


wells = pd.read_csv('train.csv', sep=';')


# In[3]:


wells


# In[4]:


len(wells.WELL.unique())


# There are 98 wells in the training dataset with more than 15 curves per well, incluiding the facies interpretation. But, before we start with the interactive visualization, we need to do some preparation to the dataset.

# ## Data preparation
# ### Lithology curve
# The facies interpretation in the dataset is contained as a 12 integer code in the 'FORCE_2020_LITHOFACIES_LITHOLOGY' column. But we will add two more columns to the dataset, one with a different integer label in order to do the visualization coding easier and the second column with a text label with the correspoding lithology name.
#

# In[5]:


# map of lithology to text label
litho_keys = {30000: 'Sandstone',
                     65030: 'Sandstone/Shale',
                     65000: 'Shale',
                     80000: 'Marl',
                     74000: 'Dolomite',
                     70000: 'Limestone',
                     70032: 'Chalk',
                     88000: 'Halite',
                     86000: 'Anhydrite',
                     99000: 'Tuff',
                     90000: 'Coal',
                     93000: 'Basement'}

# map of lithology to integer label
litho_numbers = {30000: 0,
             65030: 1,
             65000: 2,
             80000: 3,
             74000: 4,
             70000: 5,
             70032: 6,
             88000: 7,
             86000: 8,
             99000: 9,
             90000: 10,
             93000: 11}

# generation of the two new columns
wells['LITHOLOGY'] = wells['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(litho_keys)
wells['LITH_LABEL'] = wells['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(litho_numbers)


# We also need to create a colormap and color scale to display the Lithology curve in the well visualizer.

# In[6]:


# colormap of the lithologies that will be used in the cross-plot
colormap={'Sandstone': '#f4d03f',
                     'Sandstone/Shale': '#7ccc19',
                     'Shale': '#196f3d',
                     'Marl': '#160599',
                     'Dolomite': '#2756c4',
                     'Limestone': '#3891f0',
                     'Chalk': '#80d4ff',
                     'Halite': '#87039e',
                     'Anhydrite': '#ec90fc',
                     'Tuff': '#ff4500',
                     'Coal': '#000000',
                     'Basement': '#dc7633'}
#list of values and colors to create the discrete color scale for the log plot
vals = [0,1,2,3,4,5,6,7,8,9,10,11,12]
col = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc', '#FF4500', '#000000', '#DC7633']

# function to generate the discrete color scale
def discrete_colorscale(bvals, colors):
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale

dcolor = discrete_colorscale(vals,col)

# tick text that will appear next to the discrete color scale
ticktext = ['Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']

# map linking the text and integer labels for the hover information in the log plot
Litho = {0:'Sandstone', 1:'Sandstone/Shale', 2:'Shale', 3:'Marl', 4:'Dolomite', 5:'Limestone', 6:'Chalk', 7:'Halite',
 8: 'Anhydrite', 9:'Tuff', 10:'Coal', 11:'Basement'}


# ### Map information
# For the map view, we are going to create a sub-dataframe with just one well per row. But we need some information to have a better display of the wells that is not in the dataset, this is the type of well in terms of productions.
#
# We also need to convert the location coordinates from UTM to Lat/Lon, as Plotly Mapbox uses Lat/Lon coordinates.
#

# In[7]:


# map of wells to their type in terms of production
well_type = {'15/9-13':'Gas/Con', '15/9-15':'Gas/Con', '15/9-17':'Gas/Con', '16/1-2':'Oil', '16/1-6 A':'Dry',
       '16/10-1':'Dry','16/10-2':'Dry', '16/10-3':'Dry', '16/10-5':'Dry', '16/11-1 ST3':'Dry', '16/2-11 A':'Oil',
       '16/2-16':'Oil', '16/2-6':'Oil', '16/4-1':'Dry', '16/5-3':'Oil', '16/7-4':'Gas/Con', '16/7-5':'Dry',
       '16/8-1':'Dry', '17/11-1':'Dry', '25/11-15':'Oil', '25/11-19 S':'Oil/Gas', '25/11-5':'Oil',
       '25/2-13 T4':'Oil/Gas', '25/2-14':'Dry', '25/2-7':'Shows', '25/3-1':'Dry', '25/4-5':'Oil/Gas', '25/5-1':'Oil',
       '25/5-4':'Gas/Con', '25/6-1':'Oil', '25/6-2':'Dry', '25/6-3':'Dry', '25/7-2':'Gas/Con', '25/8-5 S':'Oil',
       '25/8-7':'Shows', '25/9-1':'Dry', '26/4-1':'Dry', '29/6-1':'Gas/Con', '30/3-3':'Dry', '30/3-5 S':'Oil',
       '30/6-5':'Oil', '31/2-1':'Oil/Gas', '31/2-19 S':'Dry', '31/2-7':'Dry', '31/2-8':'Shows', '31/2-9':'Oil/Gas',
       '31/3-1':'Gas', '31/3-2':'Oil/Gas', '31/3-3':'Dry', '31/3-4':'Dry', '31/4-10':'Oil/Gas', '31/4-5':'Oil',
       '31/5-4 S':'Oil/Gas', '31/6-5':'Oil/Gas', '31/6-8':'Oil/Gas', '32/2-1':'Dry', '33/5-2':'Shows', '33/6-3 S':'Dry',
       '33/9-1':'Oil', '33/9-17':'Shows', '34/10-19':'Oil', '34/10-21':'Gas', '34/10-33':'Oil',
       '34/10-35':'Gas/Con', '34/11-1':'Gas/Con', '34/11-2 S':'Gas/Con', '34/12-1':'Gas/Con', '34/2-4':'Shows',
       '34/3-1 A':'Oil', '34/4-10 R':'Oil', '34/5-1 A':'Dry', '34/5-1 S':'Oil', '34/7-13':'Oil',
       '34/7-20':'Shows', '34/7-21':'Oil', '34/8-1':'Oil/Gas', '34/8-3':'Oil/Gas', '34/8-7 R':'Gas', '35/11-1':'Dry',
       '35/11-10':'Oil/Gas', '35/11-11':'Shows', '35/11-12':'Shows', '35/11-13':'Shows', '35/11-15 S':'Oil/Gas',
       '35/11-6':'Shows', '35/11-7':'Oil/Gas', '35/12-1':'Shows', '35/3-7 S':'Gas', '35/4-1':'Shows', '35/8-4':'Dry',
       '35/8-6 S':'Dry', '35/9-10 S':'Oil/Gas', '35/9-2':'Oil/Gas', '35/9-5':'Dry', '35/9-6 S':'Oil/Gas', '36/7-3':'Dry',
       '7/1-1':'Dry', '7/1-2 S':'Dry'}


# In[8]:


# lists of wells and their location in x and y
well_x = []
well_y = []
well_lst = wells.WELL.unique()

# for loop to fill with the corresponding values of well name, x and y location.
for well in well_lst:
    well_x.append(wells.loc[wells['WELL']== well, 'X_LOC'].reset_index(drop=True)[0])
    well_y.append(wells.loc[wells['WELL']== well, 'Y_LOC'].reset_index(drop=True)[0])

# generation of the sub-dataframe by using a dictionary of the lists defined before
loc_dict = {'Well':well_lst,'Loc_x':well_x,'Loc_y':well_y}
wells_map = pd.DataFrame(loc_dict, columns=['Well', 'Loc_x', 'Loc_y'])
wells_map


# As it can be seen above, there are some wells that don't have the location. For simplicity, we are just going to remove those wells and add the well type, latitude and longitude columns.

# In[9]:


# drop of wells without a location
wells_map.dropna(inplace=True)

# add of the Well_Type column
wells_map['Well_Type'] = wells_map['Well'].apply(lambda x: well_type[x])

# calculation of the latitude and longitude of the wells by using the UTM package
wells_map['Lat'] = wells_map.apply(lambda x: utm.to_latlon(x['Loc_x'], x['Loc_y'], 31, 'V')[0], axis=1)
wells_map['Lon'] = wells_map.apply(lambda x: utm.to_latlon(x['Loc_x'], x['Loc_y'], 31, 'V')[1], axis=1)
wells_map


# ### List of wells
# For the last part of the data preparation, we are going to create a list of dictionaries of wells present in the dataset. This dictionary will be used in the dropdown menus to display an specific well in each plot.

# In[10]:


# list of dictionaries used in the dropdown menus, the format is 'label':,'value'
wells_options = [
    {"label": well, "value": well}
    for well in wells.WELL.unique()
]


# ## Dash application
# In this section we are going to generate the dash application for the well visualization. A dash app consists in two main parts, the layout of the dashboard and the callbacks of the interactive plotting.
#
# First, we are going to define the layout, that is the structure and style of the dashboard. For this we will use three main elements of dash: Dash-core-elements, Dash-html and Dash Bootstrap. If you want more information of one these elements, you can go to the [Dash User Guide](https://dash.plotly.com/).
#
# For the style of the dashboard, we are going to use the Lumen Theme from BootstrapCDN, that it is downloaded and located in the assets folder. For checking or downloading the themes of Bootstrap, you can go [here](https://www.bootstrapcdn.com/bootswatch/).

# In[11]:



app = dash.Dash(__name__)

# first we are going to define the three main bootstrap cards that will contain the menus controlling the plots.

# card with the menus to control the log plot.
card_well_log = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.P(
                        "Well to display:",
                        className="card-text",
                    ),
                    dcc.Dropdown(id='slct_well',     # dropdown menu to select which well is going to be displayed
                         options=wells_options,
                         multi=False,
                         value='15/9-13',
                         style={'width': "100%"},
                         className="dcc_control",
                         ),
                    html.Br(),
                    html.P(
                         "Curves to display:",
                         className="card-text",
                    ),
                    dcc.Checklist(id="slct_curves",    # checklist menu to select which curves are going to be displayed
                           options=[
                                    {"label": "Caliper", "value": "CALI"},
                                    {"label": "Gamma Ray", "value": "GR"},
                                    {"label": "Density", "value": "RHOB"},
                                    {"label": "Neutron", "value": "NPHI"},
                                    {"label": "Sonic P", "value": "DTC"},
                                    {"label": "Sonic S", "value": "DTS"},
                                    {"label": "Deep Res", "value": "RDEP"},
                                    {"label": "Medium Res", "value": "RMED"},
                                    {"label": "Shallow Res", "value": "RSHA"},
                                    {"label": "SP ", "value": "SP"},
                                    {"label": "Lithology", "value": "LITHOLOGY"}],
                                value=["GR",'RHOB'],
                                style={'width': "100%", 'border':'2px'},
                                labelStyle = {'display': 'block'},
                                 )
                ]
            ),
        ],
        color="ligth",
        inverse=False,
        outline=False,
        style={'height':'40rem'}
)
# card with the menus to control the map plot.
card_well_map = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.P(
                        "Filter wells to display by type:",
                        style={'margin': '0 auto'}),

                    dcc.Dropdown(id="well_types",             # dropdown menu to filter the wells by type in the map
                                    options=[
                                        {"label": "Oil", "value": "Oil"},
                                        {"label": "Gas", "value": "Gas"},
                                        {"label": "Dry", "value": "Dry"},
                                        {"label": "Gas/Condensate", "value": "Gas/Con"},
                                        {"label": "Oil/Gas", "value": "Oil/Gas"},
                                        {"label": "Shows", "value": "Shows"},],
                                     multi=True,
                                     value=['Oil','Gas','Dry','Gas/Con', 'Oil/Gas','Shows']
                    ),
                ]
            ),
        ],
        color="ligth",
        inverse=False,
        outline=False,
        style={'height':'7rem'}
)
# card with the menus to control the cross-plot.
card_well_scatt = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div([
                        html.P(
                            "Well to display:",
                            ),

                        dcc.Dropdown(id='slct_well_sct',              # dropdown menu to select the well to display
                                     options=wells_options,
                                     multi=False,
                                     value='15/9-13',
                                     ),
                        ], style={'width': "22%", 'position':'relative','left':'10px','display':'inline-block'}),
                    html.Div([
                        html.P(
                            "X-Axis:",
                            ),
                        dcc.Dropdown(id="x_axis",                      # dropdown menu to define the x-axis
                                       options=[
                                                {"label": "Caliper", "value": "CALI"},
                                                {"label": "Gamma Ray", "value": "GR"},
                                                {"label": "Neutron", "value": "NPHI"},
                                                {"label": "Density", "value": "RHOB"},
                                                {"label": "Sonic P", "value": "DTC"},
                                                {"label": "Sonic S", "value": "DTS"},
                                                {"label": "Deep Res", "value": "RDEP"},
                                                {"label": "Medium Res", "value": "RMED"},
                                                {"label": "Shallow Res", "value": "RSHA"},
                                                {"label": "SP ", "value": "SP"},
                                                {"label": "Lithology", "value": "LITHOLOGY"}],
                                            value="RHOB",
                                            multi=False
                                        ),
                            ],style={'width': "22%",  'position': 'relative', 'left':'30px', 'display':'inline-block'}),
                    html.Div([
                        html.P(
                            "Y-Axis:",
                            ),
                        dcc.Dropdown(id="y_axis",                      # dropdown menu to define the y-axis
                                       options=[
                                                {"label": "Caliper", "value": "CALI"},
                                                {"label": "Gamma Ray", "value": "GR"},
                                                {"label": "Neutron", "value": "NPHI"},
                                                {"label": "Density", "value": "RHOB"},
                                                {"label": "Sonic P", "value": "DTC"},
                                                {"label": "Sonic S", "value": "DTS"},
                                                {"label": "Deep Res", "value": "RDEP"},
                                                {"label": "Medium Res", "value": "RMED"},
                                                {"label": "Shallow Res", "value": "RSHA"},
                                                {"label": "SP ", "value": "SP"},
                                                {"label": "Lithology", "value": "LITHOLOGY"}],
                                            value="NPHI",
                                            multi=False
                                        ),
                            ],style={'width': "22%",'position': 'relative', 'left':'50px', 'display':'inline-block'}),
                    html.Div([
                        html.P(
                            "Color:",
                            ),
                        dcc.Dropdown(id="color_sct",                    # dropdown menu to define the color of the points
                                       options=[
                                                {"label": "Lithology", "value": "LITHOLOGY"},
                                                {"label": "None", "value": "None"},
                                                {"label": "Caliper", "value": "CALI"},
                                                {"label": "Gamma Ray", "value": "GR"},
                                                {"label": "Neutron", "value": "NPHI"},
                                                {"label": "Density", "value": "RHOB"},
                                                {"label": "Sonic P", "value": "DTC"},
                                                {"label": "Sonic S", "value": "DTS"},
                                                {"label": "Deep Res", "value": "RDEP"},
                                                {"label": "Medium Res", "value": "RMED"},
                                                {"label": "Shallow Res", "value": "RSHA"},
                                                {"label": "SP ", "value": "SP"}],
                                            value="LITHOLOGY",
                                            multi=False
                                        ),
                            ],style={'width': "22%",  'float': 'right', 'display':'inline-block'}),

                ]),
        ],
        color="ligth",
        inverse=False,
        outline=False,
        style={'height':'7rem'}
)

# definition of the app layout, where we are going to colocate the cards and the plots.
app.layout = html.Div([
    html.Div([html.H3("Well Visualizer",
            style={'fontSize': '35px', 'lineHeight': 1.3, 'letterSpacing': '-1px', 'marginBottom': '0px',
                   'textAlign': 'center', 'marginTop': '40px',  'fontFamily': "sans-serif"}
            ),
            html.H5("Force Dataset",
            style={'fontSize': '25px', 'lineHeight': 1.5, 'letterSpacing': '-0.5px', 'marginBottom': '20px',
                   'textAlign': 'center', 'marginTop': '0px', 'fontFamily': "sans-serif"}
            )]
    ),
    html.Div([
        dbc.Row([
            dbc.Col(card_well_log, width={'size':2, 'offset':1}),
            dbc.Col(dbc.Card([
                        html.H6("Log Plot",style={'textAlign': 'center', 'marginTop': '1rem','marginBottom': '0rem', 'fontSize': '22px', 'fontFamily': "sans-serif" }),
                        dcc.Graph(id='well_plot', figure={})], style={'height':'40rem'}
            ), width={'size':8})
        ]),
        dbc.Row([html.Br()]),
        dbc.Row([
            dbc.Col(dbc.Card([
                        html.H6("Well Map",style={'textAlign': 'center', 'marginTop': '.5rem','marginBottom': '.5rem', 'fontSize': '22px', 'fontFamily': "sans-serif" }),
                        dcc.Graph(id='well_map', figure={}, style={'marginRight':'30px'})],style={'height':'27rem'},
            ), width={'size':5, 'offset':1}),
            dbc.Col(dbc.Card([
                        html.H6("Cross-plot",style={'textAlign': 'center', 'marginTop': '.5rem','marginBottom': '.5rem', 'fontSize': '22px', 'fontFamily': "sans-serif" }),
                        dcc.Graph(id='well_scatt', figure={}, style={'marginRight':'30px', 'marginLeft':'30px'})],style={'height':'27rem'}
            ), width={'size':5})
        ]),
        dbc.Row([
                dbc.Col(card_well_map, width={'size':5, 'offset':1}),
                dbc.Col(card_well_scatt, width={'size':5})], style={'marginTop':'7px'}),
        dbc.Row([html.Br()])
    ])
], style={'background':'#f2f2f2', 'margin':0})


# Now, with the layout finished, we need to to do the app callbacks, that these are the elements that update the graphs when we are changing the dropdowns, the checklists, etc.
#
# We are going to define three callbacks for each of the three plots in our dash app.

# ### Log Plot

# In[12]:


@app.callback(
    Output(component_id='well_plot', component_property='figure'),
    [Input(component_id='slct_well', component_property='value'),
    Input(component_id='slct_curves', component_property='value')]
)

def log_plot(wll, curves):
    # generate a dataframe of the well that will be displayed
    well = wells[wells['WELL'] == wll]
    ztop=well.DEPTH_MD.min(); zbot=well.DEPTH_MD.max()
    num_curves = len(curves)

    # subplots for each of the curves that will be displayed
    fig = make_subplots(rows=1, cols=num_curves, shared_yaxes=True)

    #array of arrays of the integer label to generate the lithology curve in the log plot
    cluster=np.repeat(np.expand_dims(well['LITH_LABEL'].values,1), 100, 1)

    # generation of the hover label, so when putting the coursor over the lithology curve, the lithology is displayed
    z2=np.expand_dims(well['LITH_LABEL'].values,1)
    lst = []
    for i in z2:
        lst.append([Litho[i[0]]])
    hover = np.repeat(lst,100,1)

    # plot of each of the selected curves except for Lithology
    for ic, col in enumerate(curves):
    # if the curve doesn't exist, it will leave a blank space in the column
        if col == 'LITHOLOGY':
            continue
        if np.all(np.isnan(well[col])):
            curve = np.empty(well[col].values.shape)
            curve[:] = np.nan

        else:
            curve = well[col]

        fig.add_trace(
            go.Scatter(x=curve, y=well.DEPTH_MD, mode='lines', line=dict(width=.7),
            hovertemplate='Depth:%{y:.2f}'+'<br>Value:%{x:.2f}<extra></extra>', showlegend=False),
            row=1, col=ic+1
        )
        fig.update_yaxes(range=(zbot,ztop), row=1, col=ic+1)
        fig.update_yaxes(autorange="reversed", row=1, col=ic+1)
        fig.update_yaxes(showticklabels=False, row=1, col=ic+1)
        fig.update_xaxes(title_text=col, range=[curve.mean()*.1, curve.mean()*1.9],row=1, col=ic+1)

    # generation of the lithology curve, as we will use heatmap instead of scatter
    if 'LITHOLOGY' in curves:
        fig.add_trace(
                go.Heatmap(z=cluster, y=well.DEPTH_MD, colorscale=dcolor,zmin=-0.5,zmax=11.5,colorbar = dict(thickness=20,
                                                          tickvals=vals,
                                                          ticktext=ticktext), text=hover, hoverinfo='text'),
                row=1, col=num_curves
            )
        fig.update_yaxes(range=(zbot,ztop), row=1, col=num_curves)
        fig.update_yaxes(showticklabels=False, row=1, col=num_curves)
        fig.update_xaxes(showticklabels=False, title_text='Lithology',row=1, col=num_curves)

    # edition of the display of the figure
    fig['layout']['yaxis']['range'] = (zbot,ztop)
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['title'] = "Depth (m)"
    fig['layout']['yaxis']['showticklabels'] = True
    if len(curves) < 6:
        fig.update_layout(width=200*num_curves, height=593, showlegend=False)
    elif len(curves) > 5:
        fig.update_layout(height=593, showlegend=False)
    fig.update_layout(margin={'t':10, 'b':10, 'l':100, 'r':20}, plot_bgcolor="#F9F9F9"),
    return fig


# ### Map view

# In[13]:


@app.callback(
    Output("well_map", "figure"),
    [Input("well_types", "value")])

def well_map_plot(well_types):
    # filter of wells by the selection in the dropdown menu
    wlls = wells_map[wells_map['Well_Type'].isin(well_types)]

    # you will need to add your mapbox token to display the map
    px.set_mapbox_access_token('pk.eyJ1IjoicGFsYXZpY2kiLCJhIjoiY2tlajUwa2Y2MDF1ZjJ6bzg0Y3Ryb2Z3bSJ9.d1_EzLVQiv1ap5XuOxdhsg')

    # scatter mapbox plot of the wells colored by their well type
    fig = px.scatter_mapbox(wlls, lat="Lat", lon="Lon", color="Well_Type", labels={'Well_Type':''},
                       zoom=4, hover_data={'Well':False,'Lat':False, 'Lon':False, 'Well_Type':True},
                       hover_name='Well')

    # edition of the figure display
    fig.update_layout(legend={
        'orientation':"h",
        'yanchor':"bottom",
        'y':-.09,
        'xanchor':"right",
        'x':.87})
    fig.update_layout(height=380, margin={'b':0, 'l':30, 'r':0,'t':0})
    return fig


# ### Cross-plot

# In[14]:


@app.callback(
    Output(component_id='well_scatt', component_property='figure'),
    [Input(component_id='slct_well_sct', component_property='value'),
    Input(component_id='x_axis', component_property='value'),
    Input(component_id='y_axis', component_property='value'),
    Input(component_id='color_sct', component_property='value')]
)

def scatter_plot(wll, x_ax, y_ax, color):
    # generating a dataframe of the selected well
    well = wells[wells['WELL'] == wll]

    # we will define different plots depending of the color property
    if color == 'None':
        fig = px.scatter(well, x=x_ax, y=y_ax)
        fig.update_traces(marker={'color':'#f48c06'})
        fig.update_layout(height=380, margin={'b':0, 't':0, 'l':30, 'r':0}, plot_bgcolor="#F9F9F9")
        return fig
    elif color == 'LITHOLOGY':
        fig = px.scatter(well, x=x_ax, y=y_ax, color=color,
                         color_discrete_map=colormap)
        fig.update_xaxes(title_standoff = 3)
        fig.update_layout(legend={
        'orientation':"h",
        'yanchor':"bottom",
        'y':-.30,
        'xanchor':"right",
        'x':1})
        fig.update_layout(height=380, margin={'b':0, 't':0, 'l':30, 'r':0}, plot_bgcolor="#F9F9F9")
        return fig
    elif color == 'GR':
        fig = px.scatter(well, x=x_ax, y=y_ax, color=color,
                         color_continuous_scale='Rainbow', range_color=[0,180])
        fig.update_layout(height=380, margin={'b':0, 't':0, 'l':30, 'r':0}, plot_bgcolor="#F9F9F9")
        return fig

    elif color == 'RDEP' or color =='RMED' or color == 'RSHA':
        fig = px.scatter(well, x=x_ax, y=y_ax, color=color,
                         color_continuous_scale='Rainbow',range_color=[0,100])
        fig.update_layout(height=380, margin={'b':0, 't':0, 'l':30, 'r':0}, plot_bgcolor="#F9F9F9")
        return fig
    else:
        fig = px.scatter(well, x=x_ax, y=y_ax, color=color,
                         color_continuous_scale='Rainbow')
        fig.update_layout(height=380, margin={'b':0, 't':0, 'l':30, 'r':0}, plot_bgcolor="#F9F9F9")
        return fig


# In[15]:


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:
