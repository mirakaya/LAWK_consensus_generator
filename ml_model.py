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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score, \
	mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
import random as rd
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor


def import_files(filename): # import the csv file

	'''chunk_list = []

	for chunk in pd.read_csv(filename, sep='\t', low_memory=False):
		chunk_list.append(chunk)

	return pd.concat(chunk_list, axis=0)'''
	return pd.read_csv(filename, sep='\t')

def print_info(data): #prints data information

	#check dimensions
	data.shape

	#check the info on the columns - no null values
	data.info()

	# Summary of the numerical attributes
	data.describe()

def get_columns_with_nan_values(data):
	#get number of rows with missing values
	print(data.shape[0] - data.dropna().shape[0])

	print(data[data.isnull().any(axis=1)])

	#small number, delete rows
	data = data.dropna()

	return data

def transform_categorical_to_code(data):
	data.Virus = pd.Categorical(data.Virus)
	data["Virus"] = data.Virus.cat.codes

	data.Seq_reconstructed = pd.Categorical(data.Seq_reconstructed)
	data["Seq_reconstructed"] = data.Seq_reconstructed.cat.codes

	data.Ref_sequence = pd.Categorical(data.Ref_sequence)
	data["Ref_sequence"] = data.Ref_sequence.cat.codes

	data.Name_tool = pd.Categorical(data.Name_tool)
	data["Name_tool"] = data.Name_tool.cat.codes


	return data

def correlation(data):
	# Delete less relevant features based on the correlation with the output variable
	cor = data.corr()  # calculate correlations

	# Correlation graph
	plt.figure(figsize=(40, 40))
	sns.heatmap(cor, annot=True, cmap=sns.diverging_palette(20, 220, n=200), vmin=-1, vmax=1)
	plt.show()

	# Correlation with output variable
	cor_target = abs(cor["Actual_correctness"])
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

	data = data.drop("id", axis=1, inplace=False)

	try:
		feature_cols = data.columns
	except:
		pass

	X = data[feature_cols]  # features
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

						#TODO - implement these metrics
						ncsd = 1
						nrc = 1
						identity = 0
						model_score = 0

						info = [hidden_layer_sizes, activation, solver, alpha, learning_rate, mse, r2, mae, mape, ncsd, nrc, identity]

						print_to_files(info)







def generate_plots(data):
	data.hist(bins=50, figsize=(20, 15))
	plt.show()


if __name__ == '__main__':

	filename = "performance_model.tex"
	file_tsv = "performance_model.tsv"
	if os.path.exists(filename):
		os.remove(filename)
	if os.path.exists(file_tsv):
		os.remove(file_tsv)

	f_tsv = open(file_tsv, "w")
	f_tsv.write("hidden_layer_sizes\tactivation\tsolver\talpha\tlearning_rate\tmse\tr2\tmae\tmape\tncsd\tnrc\tidentity\n")
	f_tsv.close()


	init_time = time.perf_counter()
	data = import_files("stats.tsv")
	print("TIME ->", time.perf_counter() - init_time)
	print(data.shape)
	print(data.dtypes)

	data = transform_categorical_to_code(data)
	#correlation(data)


	X, Y = drop_columns(data)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	cross_validation_MLPRegressor(X_train, y_train, y_test)

	#data.plot(kind='scatter', x='Actual_correctness', y='Correctness_expected', color="red")
	#plt.show()

	#model = XGBRegressor(random_state=42)
	#model = KNeighborsRegressor(n_neighbors=5)
	#model = MLPRegressor(activation='relu', solver='adam', alpha=0.01)

	#model.fit(X_train, y_train)
	# Evaluate the model on the testing data
'''	y_pred = model.predict(X_test)



	print(y_pred)'''


	# Evaluate the model's performance
'''	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)
	print(f"Mean squared error: {mse:.2f}")
	print(f"R-squared: {r2:.2f}")
	print(f"Mean absolute error: {mae:.2f}")
	print(f"Mean absolute percentage error: {mape:.2f}")'''




