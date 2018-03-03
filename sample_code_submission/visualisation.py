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

class Visualisation(object):

	def __init__(self):
		"""
		Default constructor
		"""

		pass

	def plot(model, data):
		"""
		Displays a graph with data and model decision curve
		"""

		pass