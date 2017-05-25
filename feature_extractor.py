import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureExtractor(object):
	"""docstring for FeatureExtrator"""

	def __init__(self, dataset):
		super(FeatureExtractor, self).__init__()
		self.data = dataset.data
		self.data_keys = dataset.data_keys
		self.trans_data = {}
		self.transform_data()
		print self.data_keys

	def transform_data(self):
		trans_data = {}
		self.discrete_feature_key = []
		self.continous_feature_key = []
		for key in self.data_keys:
			if key.endswith('moment') or key.endswith('interval') or key.endswith('date_type') or key.endswith('vehicle_type'):
				self.discrete_feature_key.append(key)
			else:
				self.continous_feature_key.append(key)

		for key in self.data_keys:
			if key.endswith('weather'):
				#trans_data[key] = np.delete(self.data[key], [0, 1, 4], axis=1)
				trans_data[key] = self.data[key]
			elif key.endswith('moment') or key.endswith('interval') or key.endswith('date_type'):
				trans_data[key] = self.data[key]
			#elif key.endswith('object_consecutive_avg_tt'):
			#	pass
			else:
				trans_data[key] = self.data[key]
		#nonlinear transform
		self.trans_data = trans_data

	def feature_combination(self):
		#import ipdb
		#ipdb.set_trace()
		data = self.trans_data
		feature = self.feature

		X_continous_sum = reduce(lambda x, y: np.concatenate((np.sum(x, 1, keepdims=True), np.sum(y, 1, keepdims=True)), -1), \
								[self.data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_avg_tt')])
		X_continous_max = reduce(lambda x, y: np.concatenate((np.max(x, 1, keepdims=True), np.max(y, 1, keepdims=True)), -1), \
								[self.data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_avg_tt')])
		X_continous_min = reduce(lambda x, y: np.concatenate((np.min(x, 1, keepdims=True), np.min(y, 1, keepdims=True)), -1), \
								[self.data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_avg_tt')])
		X_continous_mean = reduce(lambda x, y: np.concatenate((np.mean(x, 1, keepdims=True), np.mean(y, 1, keepdims=True)), -1), \
								[self.data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_avg_tt')])
		
		weather_difference = np.array(self.data['weather']) - np.array(self.data['object_weather'])

		continous_difference = np.array(X_continous_max) - np.array(X_continous_min)
		X_new = np.concatenate((X_continous_sum, X_continous_mean, weather_difference, continous_difference),-1)

		self.new_feature = np.concatenate((X_new, feature), -1)

	def feature_selection(self):
		data = self.trans_data

		X_continous = reduce(lambda x, y: np.concatenate((x, y), -1), \
							[self.data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_avg_tt')])
		X_discrete = reduce(lambda x, y: np.concatenate((x, y), -1), \
							[self.data[key] for key in self.discrete_feature_key])
		X = np.concatenate((X_continous, X_discrete), -1)
		Y = data['object_consecutive_avg_tt']
		#X = SelectKBest(f_regression, k=15).fit_transform(X, Y)
		print X.shape
		#pca = PCA(n_components=15, svd_solver='full')
		#pca.fit(X)
		#X = pca.transform(X)
		print X.shape		
		self.feature = X
		self.target = Y
	
	def extract_simple_feature(self):
		#X = reduce(lambda x, y: np.concatenate((x, y), -1), \
		#			[self.data[key] for key in self.data_keys if not key.endswith('object_consecutive_avg_tt')])
		#Y = self.data['object_consecutive_avg_tt']
		import ipdb
		ipdb.set_trace()
		for key in self.data_keys:
			print key
		X = reduce(lambda x, y: np.concatenate((np.array(x), np.array(y)), -1), \
					[self.data[key] for key in self.data_keys if not key.endswith('object_consecutive_vol')])
		Y = self.data['object_consecutive_vol']
		return np.array(X), np.array(Y)

	def extract_feature(self):
		#import ipdb
		#ipdb.set_trace()
		self.feature_selection()
		self.feature_combination()
		print self.feature.shape
		feature = self.feature
		#poly = PolynomialFeatures(2)
		#feature = poly.fit_transform(feature)
		feature = np.concatenate((feature, self.new_feature), -1)
		#feature = normalize(self.feature)
		target = self.target
		return np.array(feature), np.array(target)


class Task2FeatureExtractor(FeatureExtractor):
	"""docstring for Task2FeatureExtractor"""
	
	def transform_data(self):
		trans_data = {}
		self.discrete_feature_key = []
		self.continous_feature_key = []
		for key in self.data_keys:
			if key.endswith('moment') or key.endswith('interval') or key.endswith('date_type'):
				self.discrete_feature_key.append(key)
			else:
				self.continous_feature_key.append(key)
		trans_data = {key: value for key, value in self.data.items() if key in self.discrete_feature_key or key in self.continous_feature_key}
		self.trans_data = trans_data

	def feature_selection(self):
		data = self.trans_data
		X_continous = reduce(lambda x, y: np.concatenate((x, y), -1), \
							[data[key] for key in self.continous_feature_key if not key.endswith('object_consecutive_vol')])
		X_discrete = reduce(lambda x, y: np.concatenate((x, y), -1), \
							[data[key] for key in self.discrete_feature_key])
		X = np.concatenate((X_continous, X_discrete), -1)
		Y = data['object_consecutive_vol']
		
		self.feature = X
		self.target = Y

	def feature_combination(self):
		pass

	def extract_feature(self):
		self.feature_selection()
		self.feature_combination()
		feature = self.feature
		target = self.target
		return np.array(feature), np.array(target)