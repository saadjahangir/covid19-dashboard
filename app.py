from datetime import datetime
from typing import Tuple
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from statsmodels.tsa import api
from pmdarima import arima
import random

pd.options.mode.chained_assignment = None
pio.templates.default = "plotly_dark"


cols_basic = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million',
              'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_tests', 'total_tests',
              'total_tests_per_thousand', 'new_tests_per_thousand', 'positive_rate', 'tests_per_case', 'people_vaccinated',
              'people_fully_vaccinated', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
              'people_fully_vaccinated_per_hundred']

locationsNotRelevant = ['Upper middle income',
                        'Low income', 'Lower middle income', 'High income']

correlationAttributes = ['total_cases', 'total_deaths',
                         'total_cases_per_million', 'total_deaths_per_million']
correlationFactors = ['total_tests', 'total_tests_per_thousand', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'total_vaccinations_per_hundred', 'total_boosters_per_hundred', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita',
                      'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index', 'excess_mortality', 'excess_mortality_cumulative', 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative_per_million']

predictionModels = ['Linear regression', 'Epsilon Support Vector Regression', 'Holt linear', 'Holt Winter', 'AR', 'MA', 'ARIMA', 'SARIMA']


rawData = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
rawData['date'] = pd.to_datetime(rawData['date'], format='%Y-%m-%d')


covidData = rawData[cols_basic]

daily = covidData.groupby("date").sum()

worldData = covidData[covidData["location"] == "World"]
worldCases = int(worldData["total_cases"].values[-1])
worldDeaths = int(worldData["total_deaths"].values[-1])
worldVaccination = int(worldData["people_vaccinated"].values[-1])


mapData = covidData[covidData['continent'].notna()]


# Helper functions for Racebars

raceData = mapData[["location", "date", 'total_cases', 'total_deaths']]
for cols in raceData.columns[2:]:
    raceData = raceData.loc[raceData[cols] >= 0]


def name_to_color(countries, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    mapping_colors = dict()

    for country in countries:
        red = random.randint(r_min, r_max)
        green = random.randint(g_min, g_max)
        blue = random.randint(b_min, b_max)
        rgb_string = 'rgb({}, {}, {})'.format(red, green, blue)

        mapping_colors[country] = rgb_string

    return mapping_colors


mapping_colors = name_to_color(raceData["location"].unique().tolist())
raceData['Color'] = raceData['location'].map(mapping_colors)

new_daily = raceData.groupby("date").sum()


def sortOrder(df, attribute):
    return df.sort_values(by=attribute, ascending=False)[:10]


def frames_animation(df, title, new_daily, attribute="total_cases"):

    list_of_frames = []

    for timestamp in new_daily.index.tolist():
        fdata = sortOrder(df[df['date'] == timestamp], attribute)
        list_of_frames.append(go.Frame(data=[go.Bar(x=fdata['location'], y=fdata[attribute],
                                                    marker_color=fdata['Color'], hoverinfo='none',
                                                    textposition='outside', texttemplate='%{x}<br>%{y}',
                                                    cliponaxis=False)],
                                       layout=go.Layout(font={'size': 14},
                                                        plot_bgcolor='#111111',
                                                        xaxis={
                                                            'showline': False, 'visible': False},
                                                        yaxis={
                                                            'showline': False, 'visible': False},
                                                        bargap=0.15,
                                                        title=title + timestamp.strftime("%d/%m/%Y"))))
    return list_of_frames


def bar_race_plot(df, title, list_of_frames, new_daily, attribute="total_cases"):
    initial_time = new_daily.index[0]
    initial_names = df[df['date'] == initial_time]["location"]
    initial_values = df[df['date'] == initial_time][attribute]
    initial_color = df[df['date'] == initial_time]["Color"]
    range_max = round(1.3*df[attribute].max())

    fig = go.Figure(
        data=[go.Bar(x=initial_names, y=initial_values,
                     marker_color=initial_color, hoverinfo='none',
                     textposition='outside', texttemplate='%{x}<br>%{y}',
                     cliponaxis=False)],
        layout=go.Layout(font={'size': 14}, plot_bgcolor='#FFFFFF',
                         xaxis={'showline': False, 'visible': False},
                         yaxis={'showline': False, 'visible': False,
                                'range': (0, range_max)},
                         bargap=0.15, title=title + initial_time.strftime("%d/%m/%Y"),
                         updatemenus=[dict(type="buttons",
                                           font={'color': "black"},
                                           buttons=[dict(label="Play",
                                                         method="animate",
                                                         args=[None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True}]),
                                                    dict(label="Stop",
                                                         method="animate",
                                                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])])]),
        frames=list(list_of_frames))

    return fig

#########################################################################################
#Ravi Correlation
#################################################################################
#Original Raw data
df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
dates = sorted(df['date'].unique())


#External Raw data
df_ext = pd.read_csv("https://raw.githubusercontent.com/Jdkong/COVID-19/main/Data/variables.csv")


#Merged data (original data + external data)
df_merged = df[df['location'].isin(df_ext['country'].tolist())].merge(df_ext,left_on='location',right_on='country').drop(columns = ['country'])

#################################################################################
#Data pre-processing
defaultMetric = 'new_cases'
df_default = df[(df['date'] == df.loc[df.index[-1],'date']) & (df['continent'].notna())]

df_continents = df[df['continent'].isnull()]


demographicOptionsV = [{'label':'Youth Population','value':'Pop_bw20_34'},{'label':'Urbanisation','value':'Urban_pop'},{'label':'Population density','value':'population_density'}]
economicOptionsV = [{'label':'GINI Index','value':'GINI_index'},{'label':'Ease of doing business','value':'Business'}]
socialOptionsV = [{'label':'Social media usage','value':'Social_media'},{'label':'Interet filtering','value':'logInternet_filtering'},{'label':'Air transport','value':'log_Air_trans'}]
allOptionsV = demographicOptionsV + economicOptionsV + socialOptionsV

InterventionOptionsS = [{'label':'Total tests per 1000','value':'total_tests_per_thousand'},{'label':'Fully-vaccinated per 100','value':'people_fully_vaccinated_per_hundred'}]
socialOptionsS = [{'label':'Stringency','value':'stringency_index'},{'label':'Hospital beds per thousand','value':'hospital_beds_per_thousand'}]
allOptionsS = InterventionOptionsS + socialOptionsS

metricOptionsV = [{'label':'Reproduction Rate','value':'reproduction_rate'}]
metricOptionsP = [{'label':'Reproduction Rate','value':'reproduction_rate'},{'label':'Total cases per million','value':'total_cases_per_million'},{'label':'Total deaths per million','value':'total_deaths_per_million'},{'label':'ICU patients per million','value':'icu_patients_per_million'},{'label':'Hospital patients per million','value':'hosp_patients_per_million'}]

#################################################################################
#Data Visualization (default)

corr1 = df_merged.groupby('date')[['GINI_index','reproduction_rate']].corr(method='pearson').unstack().iloc[:,2]

fig_corr = go.Figure(layout = go.Layout({'title':dict(text='Correction between external factors and Covid'),'margin':dict(l=60,t=40,b=100,r=220,autoexpand=False),'height':500},autosize = True)) #'margin':dict(l=10,t=10,b=10,autoexpand=False)
fig_corr.add_trace(go.Scatter(x=corr1.index,y=corr1,name='GINI',mode='lines'))


fig_corr.update_layout(
    title = {'y':0.95,'x':0.4,'xanchor':'center','yanchor':'top'},
    yaxis_title = 'Correlation factor',
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

initial_range = [
    '2020-04-01', '2020-05-10'
]

fig_corr['layout']['xaxis'].update(range=initial_range)

fig_heatmap = go.Figure(data = go.Choropleth(
                        locations = df_default['iso_code'],
                        z = df_default['new_cases'].astype(float),
                        text = df_default['location'],
                        colorscale = 'Reds',
                        marker_line_color = 'darkgray',
                        marker_line_width = 0.5,
                        colorbar_title = None,
                        colorbar=go.choropleth.ColorBar(xpad=0.09,lenmode='fraction',len=0.6)),
                layout = go.Layout({'title':dict(text='New Cases'),'margin':dict(l=10,t=10,b=10,autoexpand=False)},autosize = True,dragmode='zoom')) 



####################################################################################


app = dash.Dash(__name__)
app.title = "COVID-19 Dashboard"


app.layout = html.Div([
    html.H1("COVID-19 Dashboard"),

    html.Div(
        [
            html.Div(
                [
                    html.H2("Total Cases"),
                    html.P(f'{worldCases:,}')
                ],
                className="total-cases"
            ),

            html.Div(
                [
                    html.H2("Total Deaths"),
                    html.P(f'{worldDeaths:,}')
                ],
                className="total-deaths"
            ),

            html.Div(
                [
                    html.H2("People Vaccinated"),
                    html.P(f'{worldVaccination:,}')
                ],
                className="total-active"
            )
        ],
        className="stats-container"
    ),

    html.Div(
        [
            html.Label(
                htmlFor="category-chooser",
                children="Category"
            ),
            dcc.Dropdown(
                id="category-chooser",
                options=[
                    {
                        'label': column,
                        'value': column
                    }
                    for column in ["Evolution", "Comparison", "Correlation", "Prediction"]
                ],
                value="Evolution",
                clearable=False
            )
        ],
        className="category-container"
    ),

    html.Div(
        [
            html.Div(
                [
                    html.Label(
                        htmlFor="individual-attribute-dropdown",
                        children="Attribute"
                    ),
                    dcc.Dropdown(
                        id="individual-attribute-dropdown",
                        options=[
                            {
                                'label': column.title().replace("_", " "),
                                'value': column
                            }
                            for column in daily.columns
                        ],
                        value=daily.columns[1],
                        clearable=False
                    ),
                ],
                className="graph-map-attribute"
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id="individual-scale",
                                        options=[
                                            {
                                                'label': 'Linear',
                                                'value': 'linear'
                                            },
                                            {
                                                'label': 'Log',
                                                'value': 'log'
                                            }
                                        ],
                                        value='linear'
                                    ),

                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="individual-average",
                                                options=[{
                                                    'label': 'Show Moving Average',
                                                    'value': 'show_average'
                                                }, ],
                                                value=['show_average']
                                            ),

                                            dcc.Input(
                                                id="individual-interval",
                                                type="number",
                                                min=2,
                                                max=20,
                                                step=1,
                                                value=7
                                            ),
                                        ],
                                        className="moving-average-container"
                                    )
                                ],
                                className="individual-graph-topbar"
                            ),

                            dcc.Graph(
                                id="individual-graph"
                            ),

                            html.Div(
                                [
                                    html.Label(
                                        htmlFor="individual-country-dropdown",
                                        children="Country"
                                    ),
                                    dcc.Dropdown(
                                        id="individual-country-dropdown",
                                        options=[
                                            {
                                                'label': location,
                                                'value': location
                                            }
                                            for location in covidData['location'].unique() if location not in locationsNotRelevant
                                        ],
                                        value='Norway',
                                        clearable=False
                                    ),
                                ],
                                className="individual-graph-bottombar"
                            )
                        ],
                        className="individual-graph-container"
                    ),
                    dcc.Tabs([
                        dcc.Tab(label='Bubble Map',id='evolutionBubblemapTab',
                        children=[
                            dcc.Graph(
                                id='map',
                                hoverData={'points': [
                                    {'location': None}, {'Attribute': 'Value'}]},
                                clear_on_unhover=True),
                                
                        ]),
                        
                        dcc.Tab(label='Heat Map',id='evolutionHeatmapTab',
                        children =[html.Div(id = 'VisAreaEvolution',
                        style = {'width' : '100%','borderStyle':'solid','borderWidth':'1px'},
                                                            children = [html.Div(id = 'GraphHeatmap',style = {'width':'100%','borderStyle':'solid','borderWidth':'1px'},
                                                                                children = [dcc.Graph(id='CovidDataExplorer',hoverData={'points': [
                                                                                    {'location': None}, {'Attribute': 'Value'}]},
                                                                                clear_on_unhover=True,
                                
                                                                                                    figure=fig_heatmap)]),

                                                                        html.Div(id = 'GraphControlEvolution',style = {'width':'81%','borderStyle':'solid','borderWidth':'0px','padding':'20px','marginLeft':'10px'},
                                                                                children = [dcc.Slider(id = 'DateSelector',
                                                                                                    min = 0,
                                                                                                    max = len(df['date'].unique())-1,
                                                                                                    value = len(df['date'].unique())-1,
                                                                                                    updatemode = 'drag')
                                                                                            ]),
                                                                        html.P(id='DateToolTip',children=dates[-1],style={'width':'60px','border':'1px solid black','borderRadius':'5px','padding':'5px','position':'relative','left':'685px','top':'-50px'})]),


                        ])
                    ],
                    style={"color": "black"}),
                    
                ],
                className="graph-map-container"
            ),

            html.Div(
                [
                    html.Label(
                        id="individual-date-label",
                        htmlFor="individual-date-slider",
                    ),
                    dcc.RangeSlider(
                        id="individual-date-slider",
                        min=daily.index[0].timestamp(),
                        max=daily.index[-1].timestamp(),
                        value=[daily.index[0].timestamp(
                        ), daily.index[-1].timestamp()]
                    ),
                ],
                className="graph-map-date"
            )
        ],
        id="evolution",
        className="evolution"
    ),

    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.RadioItems(
                                id="comparison-scale",
                                options=[
                                    {
                                        'label': 'Linear',
                                        'value': 'linear'
                                    },
                                    {
                                        'label': 'Log',
                                        'value': 'log'
                                    }
                                ],
                                value='linear'
                            ),

                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="comparison-average",
                                        options=[{
                                            'label': 'Show Moving Average',
                                            'value': 'show_average'
                                        }, ],
                                        value=['show_average']
                                    ),

                                    dcc.Input(
                                        id="comparison-interval",
                                        type="number",
                                        min=2,
                                        max=20,
                                        step=1,
                                        value=7
                                    ),
                                ],
                                className="moving-average-container"
                            )
                        ],
                        className="comparison-graph-topbar"
                    ),

                    dcc.Graph(
                        id='comparison-graph'
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Countries",
                                                htmlFor="comparison-country-dropdown",
                                            ),
                                            dcc.Dropdown(
                                                id="comparison-country-dropdown",
                                                options=[
                                                    {
                                                        'label': location,
                                                        'value': location
                                                    }
                                                    for location in covidData['location'].unique() if location not in locationsNotRelevant
                                                ],
                                                multi=True,
                                                value=['Norway', "Sweden"]
                                            )
                                        ],
                                        className="comparison-country-container"
                                    ),

                                    html.Div(
                                        [
                                            html.Label(
                                                htmlFor="comparison-attribute-dropdown",
                                                children="Attribute"
                                            ),
                                            dcc.Dropdown(
                                                id="comparison-attribute-dropdown",
                                                options=[
                                                    {
                                                        'label': column,
                                                        'value': column
                                                    }
                                                    for column in daily.columns
                                                ],
                                                value=daily.columns[1],
                                                clearable=False
                                            ),
                                        ],
                                        className="comparison-attribute-container"
                                    )
                                ],
                                className="comparison-country-attribute-container"
                            ),

                            html.Div([
                                html.Label(
                                    id="comparison-date-label",
                                    htmlFor="comparison-date-slider",
                                ),
                                dcc.RangeSlider(
                                    id="comparison-date-slider",
                                    min=daily.index[0].timestamp(),
                                    max=daily.index[-1].timestamp(),
                                    value=[daily.index[0].timestamp(
                                    ), daily.index[-1].timestamp()]
                                )
                            ])
                        ],
                        className="comparison-graph-bottombar"
                    )
                ],
                className="comparison-graph-container"
            ),

            html.Div(
                [
                    dcc.Graph(
                        id='evolution-graph'
                    ),

                    html.Div([
                        html.Div(
                            [
                                html.Label(
                                    htmlFor="evolution-attribute-dropdown",
                                    children="Attribute"
                                ),
                                dcc.Dropdown(
                                    id="evolution-attribute-dropdown",
                                    options=[
                                        {
                                            'label': column,
                                            'value': column
                                        }
                                        for column in ['total_cases', 'total_deaths']
                                    ],
                                    value=daily.columns[0],
                                    clearable=False
                                ),
                            ],
                            className="evolution-attribute-container"
                        )
                    ])
                ],
                className="evolution-graph-container")
        ],
        id="comparison",
        className="comparison"
    ),

################################################################################
   

############################################################################
        html.Div(
        [
            html.Div(id='mainDiv',
                    children = [html.Div(id = 'Container',style ={'width' : '100%','borderStyle':'solid','borderWidth':'1px','display':'flex'},    
                                        children = [html.Div(id = 'VisArea',style = {'width' : '70%','borderStyle':'solid','borderWidth':'1px'},
                                                            children = [html.Div(id = 'Graph',style = {'width':'100%','borderStyle':'solid','borderWidth':'1px'},
                                                                                children = [dcc.Graph(id='CorrelationExplorer',hoverData={'points': [{'x': dates[-1]}]},
                                                                                                    figure=fig_corr,responsive=True,clear_on_unhover=False),#]),
                                                                                            dcc.Tooltip(id='CorrelationExplorerTooltip',zindex=10),

                                                                                            dcc.Checklist(id='toggleScatter',options=[{'label': 'Show scatter plot', 'value': 'checked'}],
                                                                                                        value=[],
                                                                                                        labelStyle={'display': 'inline-block'},style={'height': '200px'})]),

                                                                        html.Div(id = 'GraphControl',style = {'width':'81%','borderStyle':'solid','borderWidth':'0px','padding':'20px','marginLeft':'10px'}),                            
                                                                                            ]),

                                                    html.Div(id='UIcontainer',style={'width':'30%'},children = [
                                                           dcc.Tabs(id='Tabs',value = 'tab1',style={'color': 'black'}, children = [
                                                                dcc.Tab(id='CV',value='tab1',label='Covid vulnerability',children = [html.Div(id = 'UIArea1',style={'borderStyle':'solid','borderWidth':'1px','flexGrow':'1','display':'flex','flex-direction':'column','row-gap':'10px'},
                                                                children = [html.Div(id='COVID_Vulnerability_1',style={'width':'100%','borderStyle':'solid','borderWidth':'1px','display':'flex','flex-direction':'column','row-gap':'10px','flex-grow':'1'},
                                                                                    children=[html.Div(id='metric1', style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'}, 
                                                                                                       children = [html.Div(id='metricText1',style={'margin':'5px','width':'40%'},children=[html.P(children='Covid Metric',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                   html.Div(id='metricMenu1',style={'margin':'5px','width':'60%'},children=[dcc.Dropdown(id = 'metricSelector1',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','color':'black'},options = metricOptionsV,value='reproduction_rate', clearable=False)])]),
                                                                                        
                                                                            
                                                                                            html.Div(id='demographic1', style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'}, 
                                                                                                       children = [html.Div(id='demoText1',style={'margin':'5px','width':'40%'},children=[html.P(children='Demographic Factors',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                   html.Div(id='demoMenu1',style={'margin':'5px','width':'60%'},children=[dcc.Checklist(id = 'demographicSelector1',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px'},options = demographicOptionsV,value=['Pop_bw20_34'],labelStyle={'display':'block'})])]),
                                                                                                                                
                                                                        
                                                                                            html.Div(id='economic1',style = {'width' :'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'},
                                                                                                    children = [html.Div(id='ecoText1',style={'margin':'5px','width':'40%'},children=[html.P(children='Economic Factors',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                html.Div(id='ecoMenu1',style={'margin':'5px','width':'60%'},children=[dcc.Checklist(id = 'economicSelector1',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px'},options = economicOptionsV,value=[],labelStyle={'display':'block'})])]),
                                                                                            
                                                                                        
                                                                                            html.Div(id='social1',style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'},
                                                                                                    children = [html.Div(id='socialText1',style={'margin':'5px','width':'40%'},children=[html.P(children='Social Factos',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                html.Div(id='socialMenu1',style={'margin':'5px','width':'60%'},children=[dcc.Checklist(id = 'socialSelector1',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px',},options = socialOptionsV,value=[],labelStyle={'display':'block'})])])]),
                                                                          html.Div(id='ChartArea_1',style={'width':'100%','borderStyle':'solid','borderWidth':'1px','flex-grow':'1'},
                                                                          children = [html.Div(id='graph1',style = {'width':'100%'},
                                                                                            #children=[dcc.Graph(id='CorrelationDetails1',figure=fig_corr1,responsive=False,style={'width':'100%'},config={'autosizable':True})]
                                                                                            )]
                                                                          )])]),

                                                                dcc.Tab(id='CS',value='tab2',label='Covid performance',children = [html.Div(id = 'UIArea2',style={'borderStyle':'solid','borderWidth':'1px','flexGrow':'1','display':'flex','flex-direction':'column','row-gap':'10px'},
                                                                children = [html.Div(id='COVID_Vulnerability_2',style={'width' : '100%','borderStyle':'solid','borderWidth':'1px','display':'flex','flex-direction':'column','row-gap':'10px'},
                                                                                    children=[html.Div(id='metric2', style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'}, 
                                                                                                       children = [html.Div(id='metricText2',style={'margin':'5px','width':'40%'},children=[html.P(children='Covid Metric',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                   html.Div(id='metricMenu2',style={'margin':'5px','width':'60%'},children=[dcc.Dropdown(id = 'metricSelector2',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','color':'black'},options = metricOptionsP,value='reproduction_rate',clearable=False)])]),
                                                                                        
                                                                                            html.Div(id='Intevention2', style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'}, 
                                                                                                       children = [html.Div(id='interventionText2',style={'margin':'5px','width':'40%'},children=[html.P(children='Intervention Measures',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                   html.Div(id='interventionMenu2',style={'margin':'5px','width':'60%'},children=[dcc.Checklist(id = 'interventionSelector2',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px'},options = InterventionOptionsS,value=[],labelStyle={'display':'block'})])]),

                                                
                                                                                            html.Div(id='social2',style = {'width' : 'auto','borderStyle':'solid','borderWidth':'1px','display':'flex','justify-content':'center'},
                                                                                                    children = [html.Div(id='socialText2',style={'margin':'5px','width':'40%'},children=[html.P(children='Social Measures',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px','text-align':'center'})]),
                                                                                                                html.Div(id='socialMenu2',style={'margin':'5px','width':'60%'},children=[dcc.Checklist(id = 'socialSelector2',style = {'borderWidth':'1px','borderStyle':'solid','padding':'5px',},options = socialOptionsS,value=['stringency_index'],labelStyle={'display':'block'})])])]),
                                                                          
                                                                          html.Div(id='ChartArea_2',style={'width' : '100%','borderStyle':'solid','borderWidth':'1px','display':'flex'},
                                                                          children = [html.Div(id='graph2',style = {'width':'100%'},
                                                                                            #children=[dcc.Graph(id='CorrelationDetails2',figure=fig_corr1,responsive=True,style={'width':'100%','height':'100%'},config={'autosizable':True})]
                                                                                            )])])])
                                                               

                                                           ])]
                                                            

                                                                
                                                                                    
                                                                                    )])
                                
            ])
        ],
        id="correlation",
        className="correlation"
    ),

############################################################################

    html.Div(
        [
            html.Div(
                [
                    html.Div([
                        html.Div(
                            [
                                html.Label(
                                    htmlFor="prediction-interval",
                                    children="Prediction length (in days)"
                                ),

                                dcc.Input(
                                    id="prediction-interval",
                                    type="number",
                                    min=1,
                                    max=150,
                                    step=1,
                                    value=90
                                ),
                            ],
                            className="prediction-interval-container"
                        )
                    ]),

                    dcc.Graph(
                        id='prediction-graph'
                    ),

                    html.Div(
                        [
                            html.P(
                                "The country has no relevant data",
                                id="prediction-invalid"),

                            html.Div(
                                [
                                    html.Div([
                                        html.P("Residual sum of squares (RSS): "),
                                        html.P(id="prediction-rss")
                                    ]),
                                    html.Div([
                                        html.P("Mean absolute error (MAE): "),
                                        html.P(id="prediction-mae")
                                    ])
                                ],
                                id="prediction-scores"
                            ),

                            html.Div([
                                html.Div([
                                    html.Label(
                                        htmlFor="prediction-country-dropdown",
                                        children="Country"
                                    ),
                                    dcc.Dropdown(
                                        id="prediction-country-dropdown",
                                        options=[
                                            {
                                                'label': location,
                                                'value': location
                                            }
                                            for location in covidData['location'].unique() if location not in locationsNotRelevant and location not in covidData['continent'].dropna().unique()
                                        ],
                                        value='Norway',
                                        clearable=False
                                    )
                                ]),
                                html.Div([
                                    html.Label(
                                        htmlFor="prediction-model-dropdown",
                                        children="Model"
                                    ),
                                    dcc.Dropdown(
                                        id="prediction-model-dropdown",
                                        options=[
                                            {
                                                'label': model,
                                                'value': model
                                            }
                                            for model in predictionModels
                                        ],
                                        value=predictionModels[2],
                                        clearable=False
                                    )
                                ])
                            ])
                        ],
                        className="prediction-graph-bottombar"
                    )
                ],
                className = "prediction-graph-container"
            )
        ],
        id="prediction",
        className="prediction"
    )
], className="page")


@app.callback(
    Output('evolution', 'style'),
    Output('comparison', 'style'),
    Output('correlation', 'style'),
    Output('prediction', 'style'),
    Input('category-chooser', 'value'),
)
def updateLayout(category):
    evolutionStyle = {'display': 'none'}
    comparisonStyle = {'display': 'none'}
    correlationStyle = {'display': 'none'}
    predictionStyle = {'display': 'none'}

    if category == 'Evolution':
        evolutionStyle = {'display': 'block'}
    elif category == 'Comparison':
        comparisonStyle = {'display': 'flex'}
    elif category == 'Correlation':
        correlationStyle = {'display': 'flex'}
    elif category == 'Prediction':
        predictionStyle = {'display': 'flex'}

    return evolutionStyle, comparisonStyle, correlationStyle, predictionStyle


@app.callback(
    Output('individual-date-label', 'children'),
    [Input('individual-date-slider', 'value')],
)
def updateDateIndividual(date):
    return datetime.fromtimestamp(date[0]).strftime("%d/%m/%Y") + "->" + datetime.fromtimestamp(date[1]).strftime("%d/%m/%Y")


@app.callback(
    Output('comparison-date-label', 'children'),
    [Input('comparison-date-slider', 'value')],
)
def updateDateComparison(date):
    return datetime.fromtimestamp(date[0]).strftime("%d/%m/%Y") + "->" + datetime.fromtimestamp(date[1]).strftime("%d/%m/%Y")



@app.callback(
    Output('individual-graph', 'figure'),
    Input('individual-attribute-dropdown', 'value'),
    Input('individual-scale', 'value'),
    [Input('individual-date-slider', 'value')],
    [Input('individual-country-dropdown', 'value')],
    [Input('individual-average', 'value')],
    Input('individual-interval', 'value'),
    [Input('map', 'hoverData')],
    [Input('CovidDataExplorer', 'hoverData')]
)
def updateIndividualGraph(attribute, scale, date, country, average_check, interval_size, click_data, hover_data):
    if click_data:
        country_point = click_data['points'][0]['location']
        if country_point is not None:
            country_name = (mapData[mapData["iso_code"] == country_point])[
                "location"].values[0]
            country = country_name

    if hover_data:
        country_point = hover_data['points'][0]['location']
        if country_point is not None:
            country_name = (mapData[mapData["iso_code"] == country_point])[
                "location"].values[0]
            country = country_name

    graph = go.Figure()

    if attribute == None:
        return graph

    isLog = scale == 'log'

    dataFilteredByCountry = covidData[covidData['location'] == country]
    dataGroupedByDateAndLocation = dataFilteredByCountry.groupby(
        ['date', 'location']).sum().reset_index().set_index('date')

    dataGroupedByDateAndLocation = dataGroupedByDateAndLocation.loc[
        dataGroupedByDateAndLocation[attribute] > 0]

    dataFilteredByDate = dataGroupedByDateAndLocation.loc[datetime.fromtimestamp(
        date[0]): datetime.fromtimestamp(date[1])]

    graph.add_trace(
        go.Bar(x=dataFilteredByDate.index,
               y=dataFilteredByDate[attribute], name="Daily Values", marker_color="#636EFA")
    )
    if average_check != []:
        dataFilteredByDate[attribute] = dataFilteredByDate[attribute].rolling(
            str(interval_size)+'d').mean()

        graph.add_trace(
            go.Scatter(x=dataFilteredByDate.index,
                       y=dataFilteredByDate[attribute], mode='lines', name="Moving Average", marker_color="#636EFA")
        )

    if isLog:
        graph.update_yaxes(type="log")

    return graph


@app.callback(
    Output('map', 'figure'),
    Input('individual-attribute-dropdown', 'value'),
    [Input('individual-date-slider', 'value')]
)
def updateMap(attribute, date):
    datefilter = datetime.strptime(datetime.fromtimestamp(
        date[1]).strftime("%d/%m/%Y"), "%d/%m/%Y")

    dataGroupedByDateAndLocation = mapData.groupby(
        ['date', 'location', "iso_code", "continent"]).sum().reset_index().set_index('date')

    dataGroupedByDateAndLocation = dataGroupedByDateAndLocation.loc[
        dataGroupedByDateAndLocation[attribute] > 0]

    dataFilteredByDate = dataGroupedByDateAndLocation.loc[datefilter]

    scale = np.max(dataFilteredByDate[attribute])*0.001

    graph = go.Figure()

    dataFilteredByDate["text"] = dataFilteredByDate['location'] + '<br>' + \
        attribute + '<br>' + \
        ((dataFilteredByDate[attribute]).astype(int)).astype(str)

    graph.add_trace(
        go.Scattergeo(
            locationmode="ISO-3",
            locations=dataFilteredByDate["iso_code"],
            text=dataFilteredByDate["text"],
            marker=dict(
                size=dataFilteredByDate[attribute]/scale,
                color="#636EFA",
                sizemode='area'
            ),

            geo="geo"
        )
    )

    graph.update_layout(
        geo=go.layout.Geo(
            showcountries=True
        )
    )

    return graph


@app.callback(
    Output('comparison-graph', 'figure'),
    Input('comparison-attribute-dropdown', 'value'),
    Input('comparison-scale', 'value'),
    [Input('comparison-date-slider', 'value')],
    [Input('comparison-country-dropdown', 'value')],
    [Input('comparison-average', 'value')],
    Input('comparison-interval', 'value')
)
def updateComparisonGraph(attribute, scale, date, countries, average_check, interval_size):
    graph = go.Figure()

    if attribute == None:
        return graph

    isLog = scale == 'log'

    if average_check == []:
        dataFilteredByCountry = covidData[covidData['location'].isin(
            countries)]
        dataGroupedByDateAndLocation = dataFilteredByCountry.groupby(
            ['date', 'location']).sum().reset_index().set_index('date')
        dataGroupedByDateAndLocation = dataGroupedByDateAndLocation.loc[
            dataGroupedByDateAndLocation[attribute] > 0]
        dataFilteredByDate = dataGroupedByDateAndLocation.loc[datetime.fromtimestamp(
            date[0]): datetime.fromtimestamp(date[1])]

        graph = px.line(dataFilteredByDate, x=dataFilteredByDate.index,
                        y=attribute, log_y=isLog, color='location')
    else:
        for country in countries:
            dataFilteredByCountry = covidData[covidData['location'] == country]
            dataGroupedByDateAndLocation = dataFilteredByCountry.groupby(
                ['date', 'location']).sum().reset_index().set_index('date').sort_values(by=['location', "date"], ascending=True)

            dataFilteredByDate = dataGroupedByDateAndLocation.loc[datetime.fromtimestamp(
                date[0]): datetime.fromtimestamp(date[1])]

            dataGroupedByDateAndLocation = dataGroupedByDateAndLocation.loc[
                dataGroupedByDateAndLocation[attribute] > 0]
            dataFilteredByDate[attribute] = dataFilteredByDate[attribute].rolling(
                str(interval_size)+'d').mean()

            graph.add_trace(
                go.Scatter(x=dataFilteredByDate.index,
                           y=dataFilteredByDate[attribute], mode='lines', name=country)
            )
            if isLog:
                graph.update_yaxes(type="log")

    return graph


@app.callback(
    Output('evolution-graph', 'figure'),
    Input('evolution-attribute-dropdown', 'value'),
)
def updateEvolution(attribute):
    graph = go.Figure()
    if attribute not in ['total_cases', 'total_deaths']:
        attribute = "total_cases"
    else:
        if attribute == "total_cases":
            title = "Top 10 countries (Total Cases) "
        else:
            title = "Top 10 countries (Total Deaths) "

        list_of_frames = frames_animation(
            raceData, title, new_daily, attribute)
        graph = bar_race_plot(raceData, title,
                              list_of_frames, new_daily, attribute)
    return graph


@app.callback(
    Output('prediction-graph', 'figure'),
    Output('prediction-rss', 'children'),
    Output('prediction-mae', 'children'),
    Output('prediction-invalid', 'style'),
    Output('prediction-scores', 'style'),
    Input('prediction-country-dropdown', 'value'),
    Input('prediction-model-dropdown', 'value'),
    Input('prediction-interval', 'value')
)
def updatePredictionGraph(country, model, numberOfDays):
    attribute = 'total_cases'

    dataFilteredByCountry = rawData[rawData['location'] == country]
    dataFilteredByCountry['date_delta'] = (dataFilteredByCountry['date'] - dataFilteredByCountry['date'].min())  / np.timedelta64(1,'D')
    dataCleared = dataFilteredByCountry[['date_delta', attribute, 'date']].dropna()

    scoresStyle = {'display': 'none'}

    try:
        futureDates = pd.Series([dataFilteredByCountry['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(numberOfDays)])
        futureDatesDelta = pd.Series([dataFilteredByCountry['date_delta'].iloc[-1] + i for i in range(1, numberOfDays+1)])

        correlation = None
        prediction = None
        correlationDone = False

        if model == predictionModels[0]:
            model_linear = linear_model.LinearRegression()
            model_linear.fit(np.reshape(dataCleared['date_delta'].to_numpy(), (-1, 1)), dataCleared[attribute])

            correlation = model_linear.intercept_ + model_linear.coef_[0] * dataCleared['date_delta']
            prediction = model_linear.intercept_ + model_linear.coef_[0] * futureDatesDelta
            correlationDone = True
        elif model == predictionModels[1]:
            model_svr = svm.SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
            model_svr.fit(np.array(dataCleared['date_delta']).reshape(-1,1), np.array(dataCleared[attribute]).reshape(-1,1))

            correlation = model_svr.predict(np.array(dataCleared['date_delta']).reshape(-1,1))
            prediction = model_svr.predict(np.array(futureDatesDelta).reshape(-1,1))
            correlationDone = True
        elif model == predictionModels[2]:
            holt = api.Holt(np.asarray(dataCleared[attribute])).fit(smoothing_level=0.4, smoothing_trend=0.4, optimized=False)
            prediction = holt.forecast(len(futureDatesDelta))
        elif model == predictionModels[3]:
            es = api.ExponentialSmoothing(np.asarray(dataCleared[attribute]),seasonal_periods=14,trend='add', seasonal='mul').fit()
            prediction = es.forecast(len(futureDatesDelta))
        elif model == predictionModels[4]:
            model_ar = arima.auto_arima(dataCleared[attribute],error_action='ignore', start_p=0,start_q=0,max_p=4,max_q=0,suppress_warnings=True,stepwise=False,seasonal=False)
            model_ar.fit(dataCleared[attribute])
            prediction = model_ar.predict(len(futureDatesDelta))
        elif model == predictionModels[5]:
            model_ma= arima.auto_arima(dataCleared[attribute],error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=2,suppress_warnings=True,stepwise=False,seasonal=False)
            model_ma.fit(dataCleared[attribute])
            prediction=model_ma.predict(len(futureDatesDelta))
        elif model == predictionModels[6]:
            model_arima = arima.auto_arima(dataCleared[attribute], error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,suppress_warnings=True,stepwise=False,seasonal=False)
            model_arima.fit(dataCleared[attribute])
            prediction = model_arima.predict(len(futureDatesDelta))
        elif model == predictionModels[7]:
            model_sarima= arima.auto_arima(dataCleared[attribute], error_action='ignore',start_p=0,start_q=0,max_p=2,max_q=2,m=7,suppress_warnings=True,stepwise=True,seasonal=True)
            model_sarima.fit(dataCleared[attribute])
            prediction=model_sarima.predict(len(futureDatesDelta))
        else:
            raise "Model not choosen"

        rssError = "No values"
        maeError = "No values"
        if correlationDone:
            rssError = str(rss(dataCleared[attribute], correlation))
            maeError = str(MAE(dataCleared[attribute], correlation))
            scoresStyle = {'display': 'block'}
        
        graph = go.Figure()
        graph.update_layout(
            title="Total cases prediction for " + country + ' (' + model + ' model)',
            xaxis_title="Date",
            yaxis_title="Total cases"
        )
        graph.add_trace(go.Scatter(x=dataCleared['date'], y=dataCleared[attribute],
                        mode='lines',
                        name='Actual values'))
        if correlationDone:
            graph.add_trace(go.Scatter(x=dataCleared['date'], y=correlation,
                            mode='lines',
                            name='Correlation'))
        graph.add_trace(go.Scatter(x=futureDates, y=prediction,
                        mode='lines',
                        name='Prediction'))
        return graph, rssError, maeError, {'display': 'none'}, scoresStyle
    except:
        return go.Figure(), "", "", {'display': 'block'}, scoresStyle
def rss(y, y_hat):
    return ((y - y_hat)**2).sum()
def MAE(y, y_hat):
    n = len(y)
    differences = y - y_hat
    absolute_differences = differences.abs()
    return absolute_differences.sum() / n



#################################################################
#Correlation - Ravi - start
global metric
global factors

@app.callback(
    Output('CorrelationExplorer','figure'), #Dropdown Output

    Input('Tabs','value'),
    
    Input('demographicSelector1','value'),      #Dropdown Input    
    Input('economicSelector1','value'),        #Slider Input
    Input('socialSelector1','value'),
    Input('interventionSelector2','value'),
    Input('socialSelector2','value'),
    Input('metricSelector1','value'),
    Input('metricSelector2','value'),
    
)

def update_CorrelationGraph(tab,demo1,eco1,social1,inter2,social2,metric1,metric2):
    global metric
    global factors
    
    if tab == 'tab1':
        metric = metric1
        fig_corr.data = []
        factors = demo1 + eco1 + social1

        for factor in factors:
            correlation = df_merged.groupby('date')[[factor,metric1]].corr(method='pearson').unstack().iloc[:,2]
            fig_corr.add_trace(go.Scatter(x=correlation.index,y=correlation,name=next(item for item in allOptionsV if item['value'] == factor)['label'],mode='lines')) 
            fig_corr.update_layout(showlegend=True)
        return fig_corr

    elif tab == 'tab2':
        metric = metric2
        fig_corr.data = []
        factors = inter2 + social2
        for factor in factors:
            correlation = df_merged.groupby('date')[[factor,metric2]].corr(method='pearson').unstack().iloc[:,2]
            fig_corr.add_trace(go.Scatter(x=correlation.index,y=correlation,name=next(item for item in allOptionsS if item['value'] == factor)['label'],mode='lines'))
            fig_corr.update_layout(showlegend=True)
        return fig_corr

@app.callback(
    #Output('CorrelationDetails1','figure'),
    #Output('CorrelationDetails2','figure'),
    Output('CorrelationExplorerTooltip','show'),
    Output('CorrelationExplorerTooltip','bbox'),
    Output('CorrelationExplorerTooltip','children'),
    Input('CorrelationExplorer','hoverData'),
    Input('toggleScatter','value')
)

def update_CorrelationScatterGraph(hoverData,scatterToggle):
    
    if hoverData is None or not bool(scatterToggle):
        return False,dash.no_update,dash.no_update

    elif 'curveNumber' in hoverData['points'][0]:
        date = hoverData['points'][0]['x']
        bbox = hoverData['points'][0]['bbox']
        factor = factors[hoverData['points'][0]['curveNumber']]
        fig = go.Figure(data = go.Scatter(x=df_merged[(df_merged['date']==date)][factor],y=df_merged[df_merged['date']==date][metric],mode='markers',text=df_merged[df_merged['date']==date]['location']),layout = go.Layout({'title':dict(text='Scatter on ' + date),'margin':dict(l=50,t=40,b=50,r=30,autoexpand=False)},autosize = True))
        fig.update_layout(title={'y':0.95,'x':0.5,'xanchor':'center','yanchor':'top'},xaxis_title=next(item for item in allOptionsS+allOptionsV if item['value'] == factor)['label'],yaxis_title=next(item for item in metricOptionsV+metricOptionsP if item['value'] == metric)['label'])
            
        if 'checked' in scatterToggle:
            
            children = [
                html.Div([
                    dcc.Graph(figure=fig,responsive=True,style={'width':'100%','height':'100%'},config={'autosizable':True})
                    
                ], style={'width': '400px','height':'300px' ,'white-space': 'normal'})
            ]
        else:
            return False,dash.no_update,dash.no_update
        return True,bbox,children

    else: return False,dash.no_update,dash.no_update



##############################################################################
#lines added 

@app.callback(
    Output('CovidDataExplorer','figure'), #Dropdown Output

    Input('individual-attribute-dropdown','value'),      #Dropdown Input    
    Input('DateSelector','value'),        #Slider Input
)

def update_graph(SelectedMetric,DateSliderVal):
    SelectedDate = dates[DateSliderVal]
    df_current = df[(df['date'] == SelectedDate) & (df['continent'].notna())]
    if SelectedMetric is None:
        SelectedMetric = defaultMetric
    
    fig_heatmap.update_layout({'margin':dict(l=10,t=40,b=1,autoexpand=False)},autosize = True)
    fig_heatmap.update_traces(locations = df_current['iso_code'],
                      text=df_current['location'],
                      z=df_current[SelectedMetric].astype(float),
                      colorbar=go.choropleth.ColorBar(xpad=0.09))
    fig_heatmap.update_layout({'title':SelectedMetric.title().replace("_"," ")})
    return fig_heatmap


@app.callback(
    Output('DateToolTip','children'),
    Output('DateToolTip','style'),

    Input('DateSelector','value')
)
def display_toolTip(DateSliderValue):
    text = dates[DateSliderValue]
    maxSliderVal = len(df['date'].unique())-1
    position = str(20+(685-20)*DateSliderValue/(maxSliderVal))+'px'
    style={'width':'100px','border':'1px solid white','borderRadius':'10px','padding':'5px','position':'relative','left':position,'top':'-50px'}
    
    return text,style

#Correlation - ravi - end
#######################################################################

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8051)