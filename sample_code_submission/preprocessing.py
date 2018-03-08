"""
data preprocessing
Communication methods:
- process(data) : preprocess the data and returns a new data numpy array
"""

import numpy as np

THRESHOLD = 4.2  # Feature variance threshold

class Preprocessing(object):

	def __init__(self):
		pass

	def process(self, data):
		"""
		The data is preprocessed and useless features are erased.
		Returns the new array composed of only useful features for each
		data.
		"""

		global THRESHOLD

		data.data = self.keepOnlyFeatures(data.data, THRESHOLD)

		print("number of features deleted : {}".format(256 - len(data.data["X_train"][0])))

		return data

	def calculateVariance(self, data):
		"""
		Returns a list (u_1, u_2, ..., u_n) representing the variance
		for each feature n
		"""

		features = []
		count = 0

		# collect features vertically
		for feature in range(len(data[0])) :
			features.append([])
			for d in data:
				features[count].append(d[feature])
			count += 1

		features = np.array(features)
		result = np.array([])


		# variance
		for feature in features:
			result = np.append(result, feature.var())

		return result


	def keepOnlyFeatures(self, data, threshold):
		"""
		Suppresses features whose variance is below threshold.
		Returns the new data array
		"""

		featureVariances = self.calculateVariance(data["X_train"])
		toDelete = self.detectFeaturesToSupress(featureVariances, threshold)

		print("Feature variance threshold : {}".format(threshold))
		print("Features to delete : {}".format(toDelete))

		newData = []

		sets = ("X_train", "X_valid", "X_test")
		for indexSet in range(3):
			count = 0
			for d in data[sets[indexSet]]:
				newData.append([])
				for i in range(len(d)):
					if not i in toDelete:
						newData[-1].append(d[i])
				count += 1
			newData = np.array(newData)
			data[sets[indexSet]] = newData.copy()
			newData = []

		return data

	def detectFeaturesToSupress(self, featureVariances, threshold):
		"""
		Returns the list of features whose variance is below threshold
		"""

		toDelete = []
		index = 0
		for var in featureVariances:
			if var < threshold:
				toDelete.append(index)
			index += 1

		return toDelete
