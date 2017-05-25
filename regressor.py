import os
import json
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import gaussian_process as gp
from sklearn import kernel_ridge as kr
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt

from preprocess import preprocess, DataSet
from FeatureExtractor import FeatureExtractor

class Model(object):
	def __init__(self, args, hyper_param_dict):
		self.args = args
		self.hyper_param_dict = hyper_param_dict
		self.build_model()

	def build_model(self):
		args = self.args
		hyper_param_dict = self.hyper_param_dict
		regressor = []
		if args.model == 'lr':
			regressor.append(linear_model.LinearRegression())
			#regressor.append(linear_model.BayesianRidge())
		elif args.model == 'svm':
			regressor.append(svm.SVR())
			#regressor.append(linear_model.LinearRegression())
		elif args.model == 'ada':
			regressor.append(AdaBoostRegressor())
		elif args.model == 'lasso':
			reg = linear_model.Lasso()
			regressor.append(reg)

		elif args.model == 'ridge':
			reg = linear_model.Ridge(alpha=.5)
			regressor.append(reg)

		elif args.model == 'gpr':
			reg = gp.GaussianProcessRegressor()

		elif args.model == 'krr':
			reg = kr.KernelRidge()
			regressor.append(reg)

		elif args.model == 'knn':
			reg = KNeighborsRegressor(n_neighbors=hyper_param_dict['n_neighbors'],
										algorithm=hyper_param_dict['algorithm'],
										leaf_size=hyper_param_dict['leaf_size'],
										p=hyper_param_dict['p'])
			regressor.append(reg)
		
		elif args.model == 'dt':
			regressor.append(DecisionTreeRegressor(max_depth=3))
			#regressor.append(DecisionTreeRegressor(max_depth=5))
			#regressor.append(DecisionTreeRegressor(max_depth=8))

		elif args.model == 'rf':
			regressor.append(RandomForestRegressor())

		elif args.model == 'gb':
			regressor.append(GradientBoostingRegressor())
			'''regressor.append(GradientBoostingRegressor(loss='lad', \
														n_estimators=hyper_param_dict['n_estimators'], \
														max_depth=hyper_param_dict['max_depth'], \
														criterion=hyper_param_dict['criterion'], \
														min_samples_split=hyper_param_dict['min_samples_split'], \
														min_samples_leaf=hyper_param_dict['min_samples_leaf'], \
														min_weight_fraction_leaf=hyper_param_dict['min_weight_fraction_leaf']))
			'''
		elif args.model == 'nn':
			regressor.append(MLPRegressor(hidden_layer_sizes=hyper_param_dict['hidden_layer_sizes'],\
											solver=hyper_param_dict['solver'], \
											alpha=hyper_param_dict['alpha'], \
											max_iter=hyper_param_dict['max_iter'], \
											early_stopping=True, \
											epsilon=hyper_param_dict['epsilon']))

		elif args.model == 'en':
			#regressor.append(linear_model.LinearRegression())
			#regressor.append(linear_model.BayesianRidge())
			#regressor.append(linear_model.Lasso())
			#regressor.append(linear_model.Ridge())
			'''regressor.append(GradientBoostingRegressor(n_estimators=hyper_param_dict['n_estimators'], \
														max_depth=hyper_param_dict['max_depth'], \
														criterion=hyper_param_dict['criterion'], \
														min_samples_split=hyper_param_dict['min_samples_split'], \
														min_samples_leaf=hyper_param_dict['min_samples_leaf'], \
														min_weight_fraction_leaf=hyper_param_dict['min_weight_fraction_leaf']))
			'''
			regressor.append(GradientBoostingRegressor())
			regressor.append(GradientBoostingRegressor())
			regressor.append(RandomForestRegressor())
			regressor.append(RandomForestRegressor())
			#regressor.append(RandomForestRegressor())
			#regressor.append(KNeighborsRegressor())
			#regressor.append(AdaBoostRegressor())
			#regressor.append(DecisionTreeRegressor(max_depth=3))
			#regressor.append(DecisionTreeRegressor(max_depth=4))
			#regressor.append(DecisionTreeRegressor(max_depth=5))
			#regressor.append(DecisionTreeRegressor(max_depth=8))
			#regressor.append(DecisionTreeRegressor(max_depth=10))
		else:
			raise ValueError('invalid model')
		self.regressor = regressor


	def train(self, X, Y):
		for i in range(len(self.regressor)):
			self.regressor[i].fit(X, Y)

	def mini_batch_train(self):
		pass

	def get_param(self):
		pass
	def set_param(self):
		pass

	def predict(self, X):
		predict_result = []
		for i in range(len(self.regressor)):
			predict_result.append(self.regressor[i].predict(X))
		result = None
		#import ipdb
		#ipdb.set_trace()
		if len(predict_result) > 1:
			for i in range(len(predict_result)):
				result = predict_result[i] if result == None else result + predict_result[i]
			result = result/float(len(predict_result))
		else:
			result = predict_result[0]
		return result

	def score(self, X, Y):
		result = self.predict(X)
		rate = np.absolute(np.divide(Y - result, Y))
		scores = np.mean(rate)
		return scores

if __name__ == '__main__':
	pass
