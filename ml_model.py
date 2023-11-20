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

	return read_csv(filename, sep="\t", low_memory=False)

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

	#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	#imputer.fit(data)
	#dfaux = imputer.transform(data)

	dfaux = data.drop("Actual_correctness", axis=1, inplace=False)
	dfaux = dfaux.drop("id", axis=1, inplace=False)
	dfaux = dfaux.drop("Internal_id", axis=1, inplace=False)

	try:
		feature_cols = dfaux.columns
	except:
		pass

	X = data[feature_cols]  # features
	Y = data.Actual_correctness

	return X, Y


def fit_data_normal (model, X_train,y_train):
	# fit the model with data
	model.fit(X_train, y_train)
	name_file = "submission_data_normal.csv"
	generate_predictions(model, X_test, name_file)
	return model

def xgboost(X_train, y_train):

	model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

	#grid_clf = GridSearchCV(model, param_grid, cv=2)
	model.fit(X_train, y_train)

	name_file = "submission_xgboost.csv"
	generate_predictions(model, X_test, y_test, name_file)

	return model

def cross_validation_rand_forest(X_train, y_train):
	#param_grid = {
	#	'n_estimators': [5, 10, 15, 20, 30], #, 40, 50, 75, 100, 150, 200, 300, 500, 1000],
	#	'max_depth': [2, 5, 7, 9, 11, 17, None],
	#	'criterion': ['entropy'] #, 'gini'],
	#}

	param_grid = {
		'bootstrap': [True],
		'max_depth': [80, 90, 100, 110],
		'max_features': [2, 3],
		'min_samples_leaf': [3, 4, 5],
		'min_samples_split': [8, 10, 12],
		'n_estimators': [30, 100, 200, 300, 1000]
	}

	model = RandomForestClassifier()

	grid_clf = GridSearchCV(model, param_grid, cv=2)
	grid_clf.fit(X_train, y_train)
	print(grid_clf.best_estimator_)
	print(grid_clf.best_score_)
	print(grid_clf.best_params_)

	name_file = "submission_cv.csv"
	generate_predictions(grid_clf, X_test, y_test, name_file)

	return grid_clf

def random_forest(X_train, y_train):

	# model
	#randFor = RandomForestClassifier(criterion='entropy',n_estimators = 10, max_depth = None, random_state=seed)  #criterion entropy and gini produce very similar results
	randFor = RandomForestClassifier(criterion='entropy', max_depth=2, max_features=None, n_estimators=30)
	randFor.fit(X_train, y_train)

	#cross_validation_rand_forest(X_train, y_train)


	return randFor

def ensemble(X_train, y_train):

	#checked for parameters
	#{'activation': 'relu', 'alpha': 0.01, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999,
	# 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (5, 200, 1000), 'learning_rate': 'adaptive',
	# 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 500, 'momentum': 0.9, 'n_iter_no_change': 10,
	# 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': None, 'shuffle': True, 'solver': 'adam', 'tol': 0.0001,
	# 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}


	xgb_model = xgb.XGBClassifier(booster='gbtree', max_depth=3, min_child_weight=0.5, subsample=1)
	#xgb_model.fit(X_train, y_train)
	# Evaluate the performance of the ensemble on the testing data
	#print(f"Accuracy of the XGBClassifier: {xgb_model.score(X_test, y_test) * 100} %")

	lr = LogisticRegression(C=0.001, penalty='l1', solver='liblinear', tol=1e-05)
	#lr.fit(X_train, y_train)
	# Evaluate the performance of the ensemble on the testing data
	#print(f"Accuracy of the LogisticRegression: {lr.score(X_test, y_test) * 100} %")

	randFor = RandomForestClassifier(bootstrap= True, max_depth=20, max_features=4, n_estimators=100, min_samples_leaf=3, min_samples_split=10)
	#randFor.fit(X_train, y_train)
	# Evaluate the performance of the ensemble on the testing data
	#print(f"Accuracy of the RandomForestClassifier: {randFor.score(X_test, y_test) * 100} %")

	dtc = DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.1, splitter= "best")
	#dtc.fit(X_train, y_train)
	#print(f"Accuracy of the DecisionTreeClassifier: {dtc.score(X_test, y_test) * 100} %")


#missing
	gbc = GradientBoostingClassifier()
	#gbc.fit(X_train, y_train)
	#print(f"Accuracy of the GradientBoostingClassifier: {gbc.score(X_test, y_test) * 100} %")

	knn = KNeighborsClassifier()
	#knn.fit(X_train, y_train)
	#print(f"Accuracy of the KNeighborsClassifier: {knn.score(X_test, y_test) * 100} %")

	nn = MLPClassifier(solver="adam", alpha=0.01, hidden_layer_sizes=(5, 200, 1000), activation="relu", learning_rate="adaptive", max_iter=500)
	#nn.fit(X_train, y_train)
	# Evaluate the performance of the ensemble on the testing data
	#print(f"Accuracy of the MLPClassifier: {nn.score(X_test, y_test) * 100} %")'''



	# Combine the models using majority voting
	ensemble = VotingClassifier(estimators=[('rf', randFor), ('xgb', xgb_model), ('lr', lr), ('nn',  nn), ('dtc', dtc), ('gbc', gbc), ('knn', knn) ], voting='hard')



	# Fit the ensemble on the training data
	ensemble.fit(X_train, y_train)

	# Evaluate the performance of the ensemble on the testing data
	print(f"Accuracy of the ensemble: {ensemble.score(X_test, y_test) * 100} %")

	return ensemble

def generate_predictions(model, X_test, name_file, ids_test):

	prediction = model.predict(X_test)

	generate_submission_file(ids_test, prediction, name_file)




def generate_submission_file(ids_test, prediction, name_file):

	df = pd.DataFrame(prediction, columns=['Actual_correctness'], index=ids_test)
	df.index.name = 'ID'
	df.to_csv(name_file, index=ids_test)

def test_params_logistic_regression (X_train, y_train):
	list_penalty = ["l1","l2", "none"]
	list_solver = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
	list_tolerance = [0.0001, 0.001, 0.01, 0.1]
	list_Cs = [0.0001, 0.001, 0.01, 0.1, 1]
	model_chosen = LogisticRegression()
	score = 0
	warnings.filterwarnings("ignore")

	for curr_penalty in list_penalty:

		for curr_solver in list_solver:

			for curr_tolerance in list_tolerance:

				for curr_C in list_Cs:

					try:
						model1 = LogisticRegression(penalty=curr_penalty, dual=False, tol=curr_tolerance,
						                            C=curr_C, solver=curr_solver, max_iter=10000,
						                            multi_class='auto')

						# Fit the ensemble on the training data
						model1.fit(X_train, y_train)

						accuracy = model1.score(X_test, y_test) * 100
						# Evaluate the performance of the ensemble on the testing data
						print(f"Accuracy of the ensemble: {accuracy} %")

						if accuracy > score:
							model_chosen = model1
							score = accuracy
							print(model_chosen.get_params())

					except:
						pass

	print(model_chosen.get_params(), score)

def test_params_xgboost (X_train, y_train):
	list_penalty = ["l1","l2", "none"]
	list_solver = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
	list_tolerance = [0.0001, 0.001, 0.01, 0.1]
	list_Cs = [0.0001, 0.001, 0.01, 0.1, 1]
	model_chosen = LogisticRegression()
	score = 0
	warnings.filterwarnings("ignore")

	for curr_penalty in list_penalty:

		for curr_solver in list_solver:

			for curr_tolerance in list_tolerance:

				for curr_C in list_Cs:

					try:
						model1 = LogisticRegression(penalty=curr_penalty, dual=False, tol=curr_tolerance,
						                            C=curr_C, solver=curr_solver, max_iter=10000,
						                            multi_class='auto')

						# Fit the ensemble on the training data
						model1.fit(X_train, y_train)

						accuracy = model1.score(X_test, y_test) * 100
						# Evaluate the performance of the ensemble on the testing data
						print(f"Accuracy of the ensemble: {accuracy} %")

						if accuracy > score:
							model_chosen = model1
							score = accuracy
							print(model_chosen.get_params())

					except:
						pass

	print(model_chosen.get_params(), score)

def test_params_neural_network (X_train, y_train):

	list_activation = ["identity", "logistic", "tanh", "relu"]
	list_alpha = [0.0001, 0.001, 0.01, 0.1]
	list_solver = ["lbfgs", "sgd", "adam"]
	list_learning_rate = ["constant", "invscaling", "adaptive"]
	list_max_iterations = [1, 10, 50, 100, 200, 500, 1000]

	model_chosen = MLPClassifier()
	score = 0
	warnings.filterwarnings("ignore")

	for activation in list_activation:

		for alpha in list_alpha:

			for solver in list_solver:

				for learning_rate in list_learning_rate:

					for max_it in list_max_iterations:

						try:
							model1 = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=(5, 2), random_state=random.randint(1, 10000), activation=activation, learning_rate=learning_rate, max_iter=max_it)

							# Fit the ensemble on the training data
							model1.fit(X_train, y_train)

							accuracy = model1.score(X_test, y_test) * 100
							# Evaluate the performance of the ensemble on the testing data
							print(f"Accuracy of the ensemble: {accuracy} %")

							if accuracy > score:
								model_chosen = model1
								score = accuracy
								print(model_chosen.get_params())

						except:
							pass

	print(model_chosen.get_params(), score)

def test_params_neural_network_v2(X_train, y_train):

	list_hidden_units = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

	model_chosen = MLPClassifier()
	score = 0

	for nr_hidden_units in list_hidden_units:

		#for snd_layer_units in list_hidden_units:

		#	for trd_layer_units in list_hidden_units:


		try:

			model1 = MLPClassifier(solver="adam", alpha=0.01, hidden_layer_sizes=(nr_hidden_units, ),
			                       activation="relu", learning_rate="adaptive", max_iter=500)

			print(nr_hidden_units)
			# Fit the ensemble on the training data
			model1.fit(X_train, y_train)


			accuracy = model1.score(X_test, y_test) * 100
			print(accuracy)
			# Evaluate the performance of the ensemble on the testing data
			print(f"Accuracy of the ensemble: {accuracy} %")

			if accuracy > score:
				model_chosen = model1
				score = accuracy
				print(model_chosen.get_params())

		except:
			pass

	print(model_chosen.get_params(),"\n" , score)



def generate_plots(data):
	data.hist(bins=50, figsize=(20, 15))
	plt.show()


if __name__ == '__main__':

	data = import_files("stats.tsv")
	#print_info(data)
	#data = get_columns_with_nan_values(data)
	data = transform_categorical_to_code(data)
	#correlation(data)

	print(data.shape)

	print(data.dtypes)
	X, Y = drop_columns(data)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	#data.plot(kind='scatter', x='Actual_correctness', y='Correctness_expected', color="red")
	#plt.show()

	model = XGBRegressor(random_state=42)
	#model = KNeighborsRegressor(n_neighbors=7)

	# Initialize the MLP model
	#model = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam', alpha=0.01)

	model.fit(X_train, y_train)
	# Evaluate the model on the testing data
	y_pred = model.predict(X_test)



	print(y_pred)

	#data.plot(kind='scatter', x=X_test['Correctness_expected'], y=y_pred, color="blue")
	#plt.show()



	# Evaluate the model's performance
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	mape = mean_absolute_percentage_error(y_test, y_pred)
	print(f"Mean squared error: {mse:.2f}")
	print(f"R-squared: {r2:.2f}")
	print(f"Mean absolute error: {mae:.2f}")
	print(f"Mean absolute percentage error: {mape:.2f}")




