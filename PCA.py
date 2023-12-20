import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE

from ml_model import import_files, transform_categorical_to_code, drop_columns

if __name__ == '__main__':

	base_dataset_name = "stats_2x"

	# if pickle file exists read from there as it is faster
	if os.path.exists(base_dataset_name + '.pickle'):
		data = pd.read_pickle(base_dataset_name + '.pickle')
	else:
		data = import_files(base_dataset_name + '.tsv', base_dataset_name + '.pickle')

	print(data.shape)
	data = transform_categorical_to_code(data)

	X, y = drop_columns(data)

	# Create a PCA model to reduce the dimensionality of the data
	#pca = PCA(n_components=2)
	#X_pca = pca.fit_transform(data.data)

	# Create a t-SNE model with perplexity=30 and learning rate=500
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(X)

	# Plot the t-SNE visualization
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data.target_names)
	plt.xlabel('t-SNE 1st dimension')
	plt.ylabel('t-SNE 2nd dimension')
	plt.title('t-SNE visualization of iris dataset')
	plt.show()