"""
model visualisation
Communication methods:
- plot(model, data) : plot the model and the data
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import  KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.manifold import TSNE, Isomap
from sklearn import preprocessing

class Visualisation(object):

	def __init__(self):
		"""
		Default constructor
    		"""

    		pass

	def plot(self, model, data, predict):
		"""
		Displays a graph with data and model decision curve

		from model.py call:
			plot(self.mod, (X))

		@param model : The chosen fit model
		@param data : tuple (X, y)
		@param predict : tuple (Y) 
		"""

		x = data[0]
		y = data[1]
		y_pre = predict

		print("TSNE dimension reduction plot")
		
		### TSNE sur dataset
		
		var_tsne = TSNE(n_components=2, 
				    init='pca', perplexity=15).fit_transform(x)
		
		le = preprocessing.LabelEncoder()
		le.fit(y)
		y = le.transform(y)
		plt.figure()
		plt.scatter(var_tsne[:,0], var_tsne[:,1], c=np.array(y))
		plt.title('Dataset - TSNE')
		plt.show()
		
		### TSNE sur prediction
		
		le = preprocessing.LabelEncoder()
		le.fit(y_pre)
		y = le.transform(y_pre)
		plt.figure()
		plt.scatter(var_tsne[:,0], var_tsne[:,1], c=np.array(y_pre))
		plt.title('Predictions sur set d\'entrainement - TSNE')
		plt.show()
		
		print("Isomap dimension reduction plot")
		
		# Isomap sur dataset
		
		var_iso = Isomap(n_components=2).fit(x).fit_transform(x)
		
		le = preprocessing.LabelEncoder()
		le.fit(y)
		y = le.transform(y)
		plt.figure()
		plt.scatter(var_iso[:,0], var_iso[:,1], c=np.array(y))
		plt.title('Dataset - Isomap')
		plt.show()
		
		# Isomap sur predictions
		
		le = preprocessing.LabelEncoder()
		le.fit(y_pre)
		y = le.transform(y_pre)
		plt.figure()
		plt.scatter(var_iso[:,0], var_iso[:,1], c=np.array(y_pre))
		plt.title('Predictions sur set d\'entrainement - Isomap')
		plt.show()

		return None