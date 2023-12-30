import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ml_model import import_files, transform_categorical_to_code, drop_columns, correlation


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
		palette="crest",
		legend=True
	)

	# place legend outside center right border of plot
	plt.legend(bbox_to_anchor=(1.0, 0.75), loc='upper left', borderaxespad=2)
	plt.subplots_adjust(left=0.2, right=0.8, wspace=0.5)
	plt.title('2D PCA Graph of the Dataset')
	plt.savefig('pca_2d.pdf')
	plt.savefig('pca_2d.jpg')
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
	ax.scatter3D(pca_df[:, 0], pca_df[:, 1], pca_df[:, 2], c=y, cmap='crest', edgecolor='k')
	# Plot title of graph
	plt.title('3D PCA Graph of the Dataset')

	'''# Plot x, y, z even ticks
	ticks = np.linspace(-3, 3, num=5)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_zticks(ticks)'''

	# Plot x, y, z labels
	ax.set_xlabel('Principal Component 1', rotation=150)
	ax.set_ylabel('Principal Component 2')
	ax.set_zlabel('Principal Component 3', rotation=60)
	# place legend outside center right border of plot
	'''plt.legend(bbox_to_anchor=(1.0, 0.75), loc='upper left', borderaxespad=2)
	plt.subplots_adjust(left=0.2, right=0.8, wspace=0.5)'''
	plt.savefig('pca_3d.pdf')
	plt.savefig('pca_3d.jpg')
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






def t_sne_analysis_2D(X, y):

	# Create a t-SNE model with perplexity=30 and learning rate=500
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(X)

	data['Principal Component 1'] = X_tsne[:, 0]
	data['Principal Component 2'] = X_tsne[:, 1]

	#print('Explained variation per principal component: {}'.format(tsne.explained_variance_ratio_))

	# Plot the t-SNE visualization
	sns.set()
	# Plot 2D t-SNE Graph
	sns.scatterplot(
		x='Principal Component 1',
		y='Principal Component 2',
		data=data,
		hue=y,
		palette="crest",
		legend=True
	)

	plt.title('2D t-SNE Graph of the Dataset')
	# place legend outside center right border of plot
	plt.legend(bbox_to_anchor=(1.0, 0.75), loc='upper left', borderaxespad=2)
	plt.subplots_adjust(left=0.2, right=0.8, wspace=0.5)
	plt.savefig('t-sne_2d.pdf')
	plt.savefig('t-sne_2d.jpg')
	plt.show()

def t_sne_analysis_3D(X, y):

	# Create a t-SNE model with perplexity=30 and learning rate=500
	tsne = TSNE(n_components=3)
	X_tsne = tsne.fit_transform(X)

	data['Principal Component 1'] = X_tsne[:, 0]
	data['Principal Component 2'] = X_tsne[:, 1]
	data['Principal Component 3'] = X_tsne[:, 2]

	#print('Explained variation per principal component: {}'.format(tsne.explained_variance_ratio_))

	ax = plt.axes(projection='3d')
	ax.scatter3D(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='crest', edgecolor='k')
	# Plot title of graph
	plt.title('3D t-SNE Graph of the Dataset')

	'''# Plot x, y, z even ticks
	ticks = np.linspace(-3, 3, num=5)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_zticks(ticks)'''

	# Plot x, y, z labels
	ax.set_xlabel('Principal Component 1', rotation=150)
	ax.set_ylabel('Principal Component 2')
	ax.set_zlabel('Principal Component 3', rotation=60)
	# place legend outside center right border of plot
	'''plt.legend(bbox_to_anchor=(1.0, 0.75), loc='upper left', borderaxespad=2)
	plt.subplots_adjust(left=0.2, right=0.8, wspace=0.5)'''
	plt.savefig('t-sne_3d.pdf')
	plt.savefig('t-sne_3d.jpg')
	plt.show()

def get_boxplot(df):
	#v1
	'''plt.figure(figsize=(25, 10))
	sns.boxplot(data=data, palette="Set2")
	plt.xticks(rotation=0)
	plt.show()'''

	#v2
	'''fig = px.box(df.melt(), y="value", facet_col="variable", boxmode="overlay", color="variable")
	fig.update_yaxes(matches=None)

	for i in range(len(fig["data"])):
		yaxis_name = 'yaxis' if i == 0 else f'yaxis{i + 1}'
		fig.layout[yaxis_name].showticklabels = True

	fig.update_layout(legend=dict(bgcolor='white'))
	fig.update_layout(plot_bgcolor='white')

	fig.update_xaxes(showline=True, linewidth=2, linecolor='black')  # , mirror=True)
	fig.update_yaxes(showline=True, linewidth=2, linecolor='black')  # , mirror=True)

	fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
	fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')

	fig.show()'''

	# v3
	# Create individual boxplots
	fig, axs = plt.subplots(1, 14, sharex=True)
	axs[0].boxplot(df['Virus'])
	axs[0].set_xlabel('Virus')

	axs[1].boxplot(df['Name_tool'])
	axs[1].set_xlabel('Name_tool')

	axs[2].boxplot(df['K'])
	axs[2].set_xlabel('K')

	axs[3].boxplot(df['Seq_reconstructed'])
	axs[3].set_xlabel('Seq_reconstructed')
	axs[4].boxplot(df['Nr_A_expected'])
	axs[4].set_xlabel('Nr_A_expected')
	axs[5].boxplot(df['Nr_T_expected'])
	axs[5].set_xlabel('Nr_T_expected')
	axs[6].boxplot(df['Nr_G_expected'])
	axs[6].set_xlabel('Nr_G_expected')
	axs[7].boxplot(df['Nr_C_expected'])
	axs[7].set_xlabel('Nr_C_expected')
	axs[8].boxplot(df['Nr_N_expected'])
	axs[8].set_xlabel('Nr_N_expected')
	axs[9].boxplot(df['Correctness_expected'])
	axs[9].set_xlabel('Correctness_expected')
	axs[10].boxplot(df['Performance_list'])
	axs[10].set_xlabel('Performance_list')
	axs[11].boxplot(df['CG_content'])
	axs[11].set_xlabel('CG_content')
	axs[12].boxplot(df['CG_content'])
	axs[12].set_xlabel('CG_content')
	axs[13].boxplot(df['Actual_correctness'])
	axs[13].set_xlabel('Actual_correctness')

	fig.set_size_inches(25, 5)


	# Adjust the figure size and show
	plt.tight_layout()
	plt.savefig('boxplot.png')
	plt.savefig('boxplot.pdf')
	plt.show()





if __name__ == '__main__':

	pd.set_option('display.max_columns', 30)

	base_dataset_name = "stats"

	# if pickle file exists read from there as it is faster
	if os.path.exists(base_dataset_name + '.pickle'):
		data = pd.read_pickle(base_dataset_name + '.pickle')
	else:
		data = import_files(base_dataset_name + '.tsv', base_dataset_name + '.pickle')



	print(data.shape)
	data = transform_categorical_to_code(data)

	# correlation(data)
	#get_boxplot(data)

	data = data.sample(500000, replace=False, random_state=42)

	print(data.shape)

	#scaler = MinMaxScaler(feature_range=(0, 1))
	#data[data.columns] = scaler.fit_transform(data[data.columns])

	X, y = drop_columns(data)

	PCA_analysis_2D(X, y)
	PCA_analysis_3D(X, y)
	t_sne_analysis_2D(X, y)
	t_sne_analysis_3D(X, y)


