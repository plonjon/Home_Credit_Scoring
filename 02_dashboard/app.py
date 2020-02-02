# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objs as go

# Données client et demande d'emprunt
dash_proba = pd.read_csv("data/dash_proba.csv")
dash_application = pd.read_csv("data/dash_application.csv",
	dtype={
		'income_type': object,
		'car_owner': object,
		'home_owner': object,
		'credit_type': object,
		}
)

# Jointure données clients et probabilités de défaut
df = dash_proba.merge(dash_application, how='inner', on='client')
df.fillna(0, inplace=True)

# Nombre d'emprunts précédents
df['prev_loans'] = df['prev_extern_loans'] + df['prev_home_credit_loans']

# Création des tranches d'age
bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['18-25', '25-35', '35-45', '45-55', '55-65', '65-80']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Aggrégation par tranche d'age
df_age_group = df.groupby('age_group').agg({
        'default_proba': 'mean',
        'income': 'mean',
        'credit': 'mean',
        'annuity': 'mean',
        'duration': 'mean',})\
    .round({
        'default_proba': 2,
        'income': 0,
        'credit': 0,
        'annuity': 0,
        'duration': 2,})\
    .reset_index()

# Aggrégation par type de revenu
df_income_type = df.groupby('income_type').agg({
        'default_proba': 'mean',
        'income': 'mean',
        'credit': 'mean',
        'annuity': 'mean',
        'duration': 'mean',})\
    .round({
        'default_proba': 2,
        'income': 0,
        'credit': 0,
        'annuity': 0,
        'duration': 2,})\
    .reset_index()


# valeurs shap pour décomposition du score
shap_df = pd.read_csv('data/dash_shap_values.csv', index_col=False).set_index('SK_ID_CURR')
shap_df.index.name = 'client'

# Nombre de variables à afficher pour décomposition du score 
first_vars = 5

# Importance des variables dans le modèle
dash_feature_importances = pd.read_csv("data/dash_feature_importances.csv")

group_importances = dash_feature_importances.groupby('group_feature')\
    .agg({'importance_normalized': 'sum'})\
    .reset_index()

feature_importances = dash_feature_importances\
    .sort_values(by='importance_normalized', ascending=False)\
    .head(10)\
    .loc[:, ['feature', 'importance_normalized']]


# palette de couleurs
palette = sns.color_palette("bright", 10)

# Messages d'erreur
MSG_CLIENT = 'Saisir N° client'
MSG_ERROR = ''

# Application dash
app = dash.Dash(__name__)

server = app.server

# Feuillet principal
app.layout = html.Div(
	[
		html.Div(
			[

				html.Div(
					[
						dcc.Dropdown(
							id='liste_client',
							options=[{'label': i, 'value': i} for i in shap_df.index],
							value='',
							placeholder='N° de client',
							style={'width' : '200px'},
						),
						html.Button(id='submit_state', n_clicks=0, children="submit"),
					],
					style={'display': 'flex', 'justify-content': 'center'},
				),
				html.Div(
					[
						html.Div(
							[html.P("Client :"), html.H4(id="client_num")],
							className='text_menu',
						),
						html.Div(
							[html.P("Age :"), html.H4(id="age")],
							className='text_menu',
						),
						html.Div(
							[html.P("Type de revenu :"), html.H4(id="income_type")],
							className='text_menu',
						),
						html.Div(
							[html.P("Revenu :"), html.H4(id="income")],
							className='text_menu',
						),
					],
					style={'margin': '1px 20px 1px 20px'}
				),				
				html.Div(
					[
						dcc.Graph(id='graphe1'),
					],
					className="graph_container pretty_container",
				),
				
			],
			className="pretty_container",
			style={'display': 'flex',
				'justify-content': 'space-between',
				'flex-direction': 'column',
				'width':320,
				'padding-top': '40px'},
		),
		html.Div(
			[
				html.Div(
					[
						html.Div(
							[html.H3(id="credit_type"), html.P("Type d'emprunt"),],
							className="mini_container",
						),
						html.Div(
							[html.H3(id="credit"), html.P("Crédit"), ],
							className="mini_container",
						),
						html.Div(
							[html.H3(id="annuity"), html.P("Annuité"), ],
							className="mini_container",
						),
						html.Div(
							[html.H3(id="prev_loans"), html.P("Emprunts précédents"),],
							className="mini_container",
						),
					],
					style={'display': 'flex', 'justify-content': 'space-around'},

				),
				html.Div(
					[
						html.Div(
							[
								dcc.Graph(id='gauge_score'),
							],
							className="graph_container",
						),
						html.Div(
							[
								dcc.Graph(id='decomposition_score'),
							],
							className="graph_container",
						),						
					],
					style={'display': 'flex', 'justify-content': 'space-around'},
					className="pretty_container",
				),
				html.Div(
					[
						html.Div(
							[
								html.H4('Elément à afficher'),
								dcc.Dropdown(
									id='y_label',
									options=[
										{'label': 'Probabilité de défaut', 'value': 'default_proba'},
										{'label': 'Revenu', 'value': 'income'},
										{'label': 'Crédit', 'value': 'credit'},
										{'label': 'Annuité', 'value': 'annuity'},
										{'label': 'Durée', 'value': 'duration'},
									],
									value='default_proba',
									clearable=False,
								),
								html.H4('Regroupement'),
								dcc.Dropdown(
									id='x_label',
									options=[
										{'label': 'Age', 'value': 'age_group'},
										{'label': 'Type de revenu', 'value': 'income_type'},
									],
									value='age_group',
									clearable=False,
								),
							],
							style={'width': '200px'},
						),
						html.Div(
							[dcc.Graph(id='multi_infos_plot'),],
							className="graph_container",
						),
					],
					style={'display': 'flex', 'justify-content': 'space-around'},
					className="pretty_container",
				),
			],
			style={'display': 'flex', 'justify-content': 'space-between','flex-direction': 'column', 'flex': 1},
		),
	],
	style={'display': 'flex', 'justify-content': 'space-between'},
)

# Selection de l'identifiant client
@app.callback(
    Output("client_num", "children"),
    [Input('submit_state', 'n_clicks')],
    [State('liste_client', 'value')]
)
def update_client(n_clicks, client_id):
	
	if client_id == MSG_CLIENT:
		return ''

	try:
		client_id = int(client_id)
	except:
		return MSG_ERROR
	else:
		if any(df.client == client_id) :
			return client_id
		else :
			return MSG_ERROR


# Affichage des informations personnelles
@app.callback(
	[
		Output('age', 'children'),
		Output('income_type', 'children'),
		Output('income', 'children'),		
	],
    [
    	Input('client_num', 'children')
    ],
)
def update_client_infos(client_num):
	
	if (client_num == MSG_ERROR) | (client_num == ''):
		return '', '', ''

	thousands_sep = lambda x: '{:,.0f}'.format(x).replace(',', ' ')

	age = df.loc[df.client==int(client_num), 'age'].values[0]
	age = 'inconnu' if np.isnan(age) else age

	income_type = df.loc[df.client==int(client_num), 'income_type'].values[0]
	income_type = 'non renseigne' if pd.isnull(income_type) else income_type

	income = df.loc[df.client==int(client_num), 'income'].values[0]
	income = 'non renseigne' if np.isnan(income) else thousands_sep(income)

	return age, income_type, income


# Gauge : Affichage de la probabilité de défaut
@app.callback(
    Output("gauge_score", "figure"),
    [
    	Input('client_num', 'children'),
    ]
    
)
def update_output(client_num):
	
	default_proba = 0

	if (client_num == MSG_ERROR) | (client_num == ''):
		pass
	else:
		default_proba = df.loc[df['client'] == client_num, 'default_proba'].array[0]

	gauge_score = [
		go.Indicator(
			mode = 'gauge+number',
			value= default_proba,
			number={'suffix': "%"},
			gauge={
				'axis': {'range':[0, 100]},
				'threshold':{
					'value': 91.7936,
					'thickness': 0.9,
					'line': {'color': 'red', 'width': 3}
					},
				},		
		)
	]

	gauge_layout = {
		'autosize': False,
		'height': 250,
		'width': 300,
		'title': 'Probabilité de défaut<br>de paiement<br>',
	}
	return {
		'data': gauge_score,
		'layout': gauge_layout,
		}


# barplot horizontal : Interpretabilite du score
@app.callback(
    Output("decomposition_score", "figure"),
    [
    	Input('client_num', 'children'),
    ]
)
def update_shape_tree(client_num):
	
	default_proba = 0
	x = pd.Series(index=shap_df.iloc[0, :first_vars].index).fillna(0)

	if (client_num == MSG_ERROR) | (client_num == ''):
		pass
	else:
		default_proba = df.loc[df['client'] == client_num, 'default_proba'].array[0]
		cols = shap_df.loc[client_num]\
			.abs()\
			.sort_values(ascending=False)\
			.head(first_vars)\
			.index
		client_shap_values = shap_df.loc[client_num, cols]
		client_shap_values = client_shap_values.sort_values()
		colors= ['red' if x >= 0 else 'blue' for x in client_shap_values.values]
		x = client_shap_values

	colors= ['red' if x >= 0 else 'blue' for x in x.values]

	shape_data = [
		{
			'x': x.values,
			'y': x.index,
			'type': 'bar',
			'orientation': 'h',
			'text': np.round(x.values,3),
			'hoverinfo': 'text+y',
			'textposition': 'auto',
			'marker': {'color': colors},
			'width': 0.8,
		},
	]

	shape_layout = {
		'autosize': False,
		'margin':dict(l=300, r=20, t=50, b=50),
		'height': 250,
		'width': 550,
		'title': 'Probabilité de défaut : facteurs influents',
		'font' : {'size': 10},
		'plot_bgcolor': '#f9f9f9',
	}


	return {
		'data': shape_data,
		'layout': shape_layout,
		}

# Donut : Visualisation de l'importance des variables pour le modèle
@app.callback(
    Output("graphe1", "figure"),
    [
    	Input('client_num', 'children'),
    ]
)
def group_features(client_num):

	labels = group_importances.group_feature
	values = group_importances.importance_normalized
	colors = palette[:len(labels)]

	data_pie = [
		go.Pie(
			labels = labels,
			values= values,
			hoverinfo='label+percent',
			hole=0.5,
			textfont={'size': 10},
			marker={'colors': colors, 'line': {'color': '#000000', 'width': 2}},
		)
	]
	layout_pie = {
		'autosize': True,
		'title':'Informations prises en compte<br>dans le calcul de la probabilité',
		'titlefont': {'size': 14},
		'legend': {'x': -0.5, 'y': 0, 'orientation': 'h', 'font': {'size': 10}, 'xanchor':'left', 'yanchor': 'top'},
		'height': 350,
	}
	return {
		'data': data_pie,
		'layout': layout_pie,
		}


# Affichage des informations de l'emprunt
@app.callback(
	[
		Output('credit_type', 'children'),
		Output('credit', 'children'),
		Output('annuity', 'children'),
		Output('prev_loans', 'children'),
	],
    [
    	Input('client_num', 'children')
    ],
)
def update_client_infos(client_num):
	
	if (client_num == MSG_ERROR) | (client_num == ''):
		return '__', '__', '__', '__'

	thousands_sep = lambda x: '{:,.0f}'.format(x).replace(',', ' ')

	credit_type = df.loc[df.client==int(client_num), 'credit_type'].values[0]
	credit_type = 'inconnu' if pd.isnull(credit_type) else credit_type

	credit = df.loc[df.client==int(client_num), 'credit'].values[0]
	credit = 'non renseigne' if np.isnan(credit) else thousands_sep(credit)

	annuity = df.loc[df.client==int(client_num), 'annuity'].values[0]
	annuity = 'inconnu' if np.isnan(annuity) else thousands_sep(annuity)

	prev_loans = df.loc[df.client==int(client_num), 'prev_loans'].values[0]
	prev_loans = 'inconnu' if np.isnan(prev_loans) else prev_loans

	return credit_type, credit, annuity, prev_loans


# Barplots + line : Visualisation des informations client / regroupement
@app.callback(
    Output("multi_infos_plot", "figure"),
    [
    	Input('client_num', 'children'),
    	Input('y_label', 'value'),
    	Input('x_label', 'value'),
    ]
)
def barplot(client_num, y_label, x_label):

	if x_label == 'age_group':
		df_group = df_age_group
	elif x_label == 'income_type':
		df_group = df_income_type
	else:
		pass

	labels = df_group[x_label]
	values = df_group[y_label]

	if (client_num == MSG_ERROR) | (client_num == ''):
		client_value = np.nan
	else:
		client_value = df.loc[df.client==int(client_num), y_label].values[0]
		client_value = np.nan if np.isnan(client_value) else client_value

	axis_labels = {
	    'default_proba': 'Probabilité de défaut',
	    'income': 'Revenu',
	    'credit': 'Crédit',
	    'annuity': 'Annuité',
	    'duration': 'Durée',
	    'age_group': 'par tranche d\'age',
	    'income_type': 'par type de revenu',
	}

	multi_bar = [
		{
			'x': labels,
			'y': values,
			'type': 'bar',
			'name': 'Groupe',
			'width': 0.5,
		},
		{
			'x': labels,
			'y': [client_value for x in labels],
			'type': 'line',
			'mode': 'lines',
			'name': 'Client',
		}
	]

	multi_layout = {
		'autosize': False,
		'automargin': False,
		'hoverinfo': 'y',
		'height': 300,
		'width': 550,
		'yaxis': {'title': y_label},
		'xaxis': {'title': x_label},
		'plot_bgcolor': '#f9f9f9',
		'title': axis_labels[y_label]+' '+axis_labels[x_label],
	}
	return {
		'data': multi_bar,
		'layout': multi_layout,
		}

if __name__ == '__main__':
    app.run_server(debug=True)