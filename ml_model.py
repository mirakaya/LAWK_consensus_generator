import pickle
import time
import warnings

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as  plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score, \
	mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import random as rd
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor




def import_files(filename, name_pickle): # import the csv file

	chunks = pd.read_csv(filename, sep='\t', low_memory=False, chunksize=500000)
	data = pd.concat(chunks, ignore_index=True)

	print(data)

	#drop ref columns
	data = data.drop('Ref_sequence', axis=1)
	data = data.drop('Nr_A_ref', axis=1)
	data = data.drop('Nr_T_ref', axis=1)
	data = data.drop('Nr_C_ref', axis=1)
	data = data.drop('Nr_G_ref', axis=1)
	data = data.drop('Nr_N_ref', axis=1)
	data = data.drop('id', axis=1)

	#Normalize columns
	data["Nr_A_expected"] = data['Nr_A_expected'] / data['K']
	data["Nr_C_expected"] = data['Nr_C_expected'] / data['K']
	data["Nr_T_expected"] = data['Nr_T_expected'] / data['K']
	data["Nr_G_expected"] = data['Nr_G_expected'] / data['K']
	data["Nr_N_expected"] = data['Nr_G_expected'] / data['K']

	#add cg content
	data["CG_content"] = data['Nr_C_expected'] + data['Nr_G_expected']
	data["AT_content"] = data['Nr_A_expected'] + data['Nr_T_expected']

	#drop -1 values
	data = data.drop(data[data['Actual_correctness'] == -1].index)

	#rearrange columns
	data = data[['Virus', 'Name_tool', 'K', 'Seq_reconstructed',
	             'Nr_A_expected', 'Nr_T_expected', 'Nr_C_expected', 'Nr_G_expected', 'Nr_N_expected',
	             'Correctness_expected', 'Performance_list', 'CG_content', 'AT_content', 'Actual_correctness']]

	data.to_pickle(name_pickle)


	return data




def print_info(data): #prints data information

	#check dimensions
	data.shape

	#check the info on the columns - no null values
	data.info()

	# Summary of the numerical attributes
	data.describe()



def transform_categorical_to_code(data):

	data.replace({'Virus': {'B19V': 1, 'HPV68': 2, 'VZV': 3, 'MCPyV': 4},
	              'Name_tool': {"coronaspades": 1, "haploflow": 2, "lazypipe":3, "metaspades":4,
	                            "metaviralspades":5, "pehaplo":6, "qure":7, "qvg":8, "spades":9,
	                            "ssake":10, "tracespipe":11, "tracespipelite":12, "v-pipe":13,
	                            "virgena":14, "vispa":15}}, inplace=True)

	data['Seq_reconstructed'] = data['Seq_reconstructed'].replace(to_replace ='[aA]', value = '1', regex = True)
	data['Seq_reconstructed'] = data['Seq_reconstructed'].replace(to_replace='[cC]', value='2', regex=True)
	data['Seq_reconstructed'] = data['Seq_reconstructed'].replace(to_replace='[tT]', value='3', regex=True)
	data['Seq_reconstructed'] = data['Seq_reconstructed'].replace(to_replace='[gG]', value='4', regex=True)
	data['Seq_reconstructed'] = data['Seq_reconstructed'].replace(to_replace='[nN]', value='0', regex=True)


	data.astype({'Performance_list': 'int64'})
	data=data.apply(pd.to_numeric, errors='coerce')

	'''data.Virus = pd.Categorical(data.Virus)
	data["Virus"] = data.Virus.cat.codes

	data.Name_tool = pd.Categorical(data.Name_tool)
	data["Name_tool"] = data.Name_tool.cat.codes'''

	'''data.Seq_reconstructed = pd.Categorical(data.Seq_reconstructed)
	data["Seq_reconstructed"] = data.Seq_reconstructed.cat.codes'''

	'''data.Performance_list = pd.Categorical(data.Performance_list)
	data["Performance_list"] = data.Performance_list.cat.codes'''

	'''data.Ref_sequence = pd.Categorical(data.Ref_sequence)
	data["Ref_sequence"] = data.Ref_sequence.cat.codes'''




	return data

def correlation(data):
	# Delete less relevant features based on the correlation with the output variable
	cor = data.corr()  # calculate correlations

	sns.set(font_scale=1.3)

	# Correlation graph
	plt.figure(figsize=(22, 22))
	sns.heatmap(cor, annot=True, cmap=sns.diverging_palette(20, 220, n=200), vmin=-1, vmax=1)
	plt.savefig('correlation.pdf')
	plt.savefig('correlation.jpg')
	plt.show()


	# Correlation with output variable
	#cor_target = abs(cor["Actual_correctness"])
	#list_columns_dropped = remove_low_correlation(data, cor_target)

	#return list_columns_dropped

def remove_low_correlation(data, cor_target):

	aux = 0

	list_columns_dropped = []

	print(data.shape)
	for i in data.columns:

		try:
			if cor_target[aux] < 0.1 and i != "fraud_bool" and aux < 32:  # if two features are too different or too similar to each other, they don't carry information

				data.drop(i, axis=1, inplace=True)
				list_columns_dropped.append(i)
		except:
			pass
		aux += 1

	print(data.shape)

	return list_columns_dropped

def drop_columns(data):

	X = data.drop("Actual_correctness", axis=1).values
	Y = data.Actual_correctness

	return X, Y


def print_to_files(info):

	f_tex = open("performance_model.tex", "a")
	f_tsv = open("performance_model.tsv", "a")

	for elem in info:
		i = str(elem)
		f_tex.write(i + " & ")
		f_tsv.write(i + "\t")
	f_tex.write("\\\\\\hline\n")
	f_tsv.write("\n")

	f_tex.close()
	f_tsv.close()


def cross_validation_MLPRegressor(X_train, y_train, y_test):
	param_activation = ['tanh', 'relu']
	param_solver = ['sgd', 'adam']
	param_alpha= [0.0001, 0.001, 0.01, 0.1]
	param_learning_rate = ['constant', 'adaptive']
	param_hidden_layer_sizes = [(50, 50, 50), (50, 100, 50), (100,), (5, 5, 5), (50, 50), (5, 5), (2, 5, 2)]

	for activation in param_activation:

		for solver in param_solver:

			for alpha in param_alpha:

				for learning_rate in param_learning_rate:

					for hidden_layer_sizes in param_hidden_layer_sizes:

						model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, random_state=42)

						model.fit(X_train, y_train)
						y_pred = model.predict(X_test)

						# Evaluate the model's performance
						mse = mean_squared_error(y_test, y_pred)
						r2 = r2_score(y_test, y_pred)
						mae = mean_absolute_error(y_test, y_pred)
						mape = mean_absolute_percentage_error(y_test, y_pred)
						print(hidden_layer_sizes, activation, solver, alpha, learning_rate)
						print(f"Mean squared error: {mse:.2f}")
						print(f"R-squared: {r2:.2f}")
						print(f"Mean absolute error: {mae:.2f}")
						print(f"Mean absolute percentage error: {mape:.2f}")



						info = [hidden_layer_sizes, activation, solver, alpha, learning_rate, mse, r2, mae, mape]

						print_to_files(info)

def cross_validation_MLPRegressor_v2(X_train, y_train, y_test):
	param_grid = {
		#'hidden_layer_sizes': [(20,), (15, 30), (15, 15), (10,), (10,10), (20,10), (10,20), (20,20)],
		'hidden_layer_sizes': [(20, 20), (20, 20, 20), (20, 20, 20, 20)],
		#'hidden_layer_sizes': [(20,), (19,), (18,), (21,), (22,), (20, 5), (20, 20), (20, 30)],
		'activation': ['relu'],
		'solver': ['adam', 'sgd'],
		'alpha': [0.01, 0.005, 0.05],
		'learning_rate': ['constant', 'adaptive']
	}

	model = MLPRegressor(random_state=42, early_stopping = True)

	cv = GridSearchCV(model, param_grid, n_jobs=6, verbose=10, cv=3)
	cv.fit(X_train, y_train)
	print(cv.best_estimator_)
	print(cv.best_score_)
	print(cv.best_params_)

	return cv

def cross_validation_GradientBoostingRegression(X_train, y_train, y_test):

	param_grid = {
		'loss': ["squared_error", "absolute_error"],
		'learning_rate': [0.1, 0.2, 0.3],
		'criterion': ["friedman_mse"],
		'n_estimators': [15, 30, 50],
		'min_samples_split': [2, 4]
	}

	model = GradientBoostingRegressor(random_state=42)

	cv = GridSearchCV(model, param_grid, n_jobs=6, verbose=10, cv=3)
	cv.fit(X_train, y_train)
	print(cv.best_estimator_)
	print(cv.best_score_)
	print(cv.best_params_)

	return cv

def cross_validation_NNR(X_train, y_train, y_test):

	param_grid = {
		'n_neighbors': [1, 3, 5, 7, 9, 15],
		'weights': ['uniform', 'distance'],
		'algorithm': ['brute', 'auto'],
	}

	model = KNeighborsRegressor()

	cv = GridSearchCV(model, param_grid, n_jobs=-1, verbose=10, cv=2, error_score='raise')
	cv.fit(X_train, y_train)
	print(cv.best_estimator_)
	print(cv.best_score_)
	print(cv.best_params_)

	return cv



def cross_validation_LinearRegression(X_train, y_train, y_test):

	param_grid = {
		'fit_intercept': [True, False],
		'n_jobs': [-1],
		'C': [1, 5, 10, 50, 100],
		'gamma': ["scale", "auto"]
	}

	model = LinearRegression(random_state=42)

	cv = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1, cv=2)
	cv.fit(X_train, y_train)
	print(cv.best_estimator_)
	print(cv.best_score_)
	print(cv.best_params_)

	return cv

def generate_plots(data):
	data.hist(bins=50, figsize=(20, 15))
	plt.show()


def fit_and_predict(model, name):

	print("Testing the " + name + "...")

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	# Evaluate the model's performance
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)
	print(f"Mean squared error: {mse:.2f}")
	print(f"R-squared: {r2:.2f}")
	print(f"Mean absolute error: {mae:.2f}")
	print(f"Mean absolute percentage error: {mape:.2f}")

	# save model
	filename = name + '.sav'
	pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':

	pd.set_option('display.max_columns', 30)

	filename = "performance_model.tex"
	file_tsv = "performance_model.tsv"
	if os.path.exists(filename):
		os.remove(filename)
	if os.path.exists(file_tsv):
		os.remove(file_tsv)

	f_tsv = open(file_tsv, "w")
	f_tsv.write("hidden_layer_sizes\tactivation\tsolver\talpha\tlearning_rate\tmse\tr2\tmae\tmape\n")
	f_tsv.close()


	init_time = time.perf_counter()

	base_dataset_name = "stats"

	#if pickle file exists read from there as it is faster
	if os.path.exists(base_dataset_name + '.pickle'):
		data = pd.read_pickle(base_dataset_name + '.pickle')
	else:
		data = import_files(base_dataset_name + '.tsv', base_dataset_name + '.pickle')


	print("TIME ->", time.perf_counter() - init_time)
	print(data.shape)


	data = transform_categorical_to_code(data)
	print(data.dtypes)
	print(data.shape)
	#
	correlation(data)

	# saving as tsv file
	#data.to_csv('example.tsv', sep="\t")

	X, Y = drop_columns(data)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	'''print("Starting gradientboosting")
	cross_validation_GradientBoostingRegression(X_train, y_train, X_test)
'''
	'''print("Starting MLPRegressor")
	model_mlp = cross_validation_MLPRegressor_v2(X_train, y_train, X_test)
'''


	mlp_model = MLPRegressor(activation='relu', alpha=0.01, early_stopping=True, learning_rate='constant', hidden_layer_sizes=(20, 20), solver='adam', random_state = 42)

	fit_and_predict(mlp_model, "mlp_model")

	'''Mean squared error: 0.02
R-squared: 0.72
Mean absolute error: 0.05
Mean absolute percentage error: 36661456515607.19'''


	gbr_model = GradientBoostingRegressor(learning_rate=0.3,n_estimators=50, random_state=42)

	fit_and_predict(gbr_model, "gbr_model")
	
	'''Mean squared error: 0.01
R-squared: 0.84
Mean absolute error: 0.03
Mean absolute percentage error: 25563785680183.20'''



