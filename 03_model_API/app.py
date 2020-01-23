# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("data/model.sav", "rb"))
df = pd.read_csv("data/sample_input.csv").set_index("SK_ID_CURR")
clients = list(df.index)


# Rappel : serveur renvoie du code HTML au client, interprété par le navigateur
# Décorateur qui associe une URL à la fonction 
@app.route("/")
def home():
	return render_template("page_predict.html", clients=clients)

@app.route("/predict", methods=['POST'])
def predict():
	output_list = [x for x in request.form.values()]

	try:
		output = int(output_list[0])
	except:
		return render_template('page_predict.html', clients=clients, id_client="", prediction="")
	else:
		output = int(output_list[0])

	prob = model.predict(df.loc[[output]])[0] * 100
	id_client = 'Customer ID {}'.format(output)
	prediction = 'Default Probability : {:.2f} %'.format(prob)
	return render_template('page_predict.html', clients=clients, id_client=id_client, prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)