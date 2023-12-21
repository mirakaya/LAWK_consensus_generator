import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

from ml_model import import_files, transform_categorical_to_code, drop_columns

def PCA_analysis_2D(X, y):

	pca = PCA(n_components=2)
	pca_df = pca.fit_transform(X)

	#pca_df = pca_df[100000,:]

	data['Principal Component 1'] = pca_df[:, 0]
	data['Principal Component 2'] = pca_df[:, 1]

	print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

	#sns.scatterplot(x="pca-one", y="pca-two", data=data)
	'''plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="pca-one", y="pca-two",
		palette=sns.color_palette("hls", 10),
		data=data.sample(n=100000),
		legend="full",
		alpha=0.3
	)
	
	plt.show()'''


	# Plot the results
	'''plt.scatter(pca_df[:, 0], pca_df[:, 1], c=y, cmap='viridis', edgecolor='k')
	plt.title('PCA of the dataset')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.show()'''



	sns.set()
	# Plot 2D PCA Graph
	sns.scatterplot(
		x='Principal Component 1',
		y='Principal Component 2',
		data=data,
		hue=y,
		legend=True
	)

	plt.title('2D PCA Graph of Iris Dataset')
	plt.show()

def PCA_analysis_3D(X, y):
	pca = PCA(n_components=3)
	pca_df = pca.fit_transform(X)
	print(type(pca_df))

	# pca_df = pca_df[100000,:]

	data['Principal Component 1'] = pca_df[:, 0]
	data['Principal Component 2'] = pca_df[:, 1]
	data['Principal Component 3'] = pca_df[:, 2]

	print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

	# sns.scatterplot(x="pca-one", y="pca-two", data=data)
	'''plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="pca-one", y="pca-two",
		palette=sns.color_palette("hls", 10),
		data=data.sample(n=100000),
		legend="full",
		alpha=0.3
	)

	plt.show()'''

	# Plot the results
	ax = plt.axes(projection='3d')
	ax.scatter3D(pca_df[:, 0], pca_df[:, 1], pca_df[:, 2], c=y, cmap='viridis', edgecolor='k')
	# Plot title of graph
	plt.title(f'3D Scatter of Iris')

	'''# Plot x, y, z even ticks
	ticks = np.linspace(-3, 3, num=5)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_zticks(ticks)'''

	# Plot x, y, z labels
	ax.set_xlabel('sepal_length', rotation=150)
	ax.set_ylabel('sepal_width')
	ax.set_zlabel('petal_length', rotation=60)
	plt.show()

	'''sample = data.sample(n=100000, replace=False)
	sns.set()
	# Plot 2D PCA Graph
	sns.scatterplot(
		x='Principal Component 1',
		y='Principal Component 2',
		data=sample,
		hue=y,
		fit_reg=False,
		legend=True
	)

	plt.title('2D PCA Graph of Iris Dataset')
	plt.show()'''






def t_sne_analysis(X):

	# Create a t-SNE model with perplexity=30 and learning rate=500
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(X)

	# Plot the t-SNE visualization
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data.target_names)
	plt.xlabel('t-SNE 1st dimension')
	plt.ylabel('t-SNE 2nd dimension')
	plt.title('t-SNE visualization of iris dataset')
	plt.show()


if __name__ == '__main__':

	pd.set_option('display.max_columns', 30)

	base_dataset_name = "stats_2x"

	# if pickle file exists read from there as it is faster
	if os.path.exists(base_dataset_name + '.pickle'):
		data = pd.read_pickle(base_dataset_name + '.pickle')
	else:
		data = import_files(base_dataset_name + '.tsv', base_dataset_name + '.pickle')

	print(data.shape)
	data = transform_categorical_to_code(data)

	print(data.describe())

	X, y = drop_columns(data)

	PCA_analysis_2D(X, y)
	#PCA_analysis_3D(X, y)
	#t_sne_analysis(X)


