import os
import json
import argparse
import numpy as np
from regressor import Model
#from evalutor import LabelEvalutor
from preprocess import DataSet
from preprocess import preprocess as travel_time_preprocess
from FeatureExtractor import FeatureExtractor, Task2FeatureExtractor
from volume_preprocess import volume_preprocess
from submission import submit, volume_submit
from HyperParameterTunning import HyperParameterTunning
from utils import pprint

path = '../data/dataSets/training/'
suffix = '.json'

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="svm", help="designated model will be used to do regression")
	parser.add_argument("--task", type=int, default=1, help="")
	args = parser.parse_args()
	return args

def main():
	#import ipdb
	#ipdb.set_trace()
	args = get_args()

	if args.task == 1:
		preprocess = travel_time_preprocess
		data_file = 'train_data_for_training'
		validation_data_file = 'train_data_for_validation'
	else:
		preprocess = volume_preprocess
		data_file = 'task2_train_data_for_training'
		validation_data_file = 'task2_train_data_for_validation'

	if args.model == 'en':
		return args, None
	else:
		train_set, _ = preprocess(data_file, rate=0.0)
		dev_set, _ = preprocess(validation_data_file, rate=0.0)
		hpt = HyperParameterTunning(args)
		hpt.tune_parameter([train_set, dev_set])
		p, s = hpt.get_best_pamater()
		pprint(p)
		print s
		return args, p
	'''
	model = Model(args)
	train_set, dev_set = preprocess(rate=0.0)
	test_set, _ = preprocess(data_file='test_data', rate=0.0)
	fe = FeatureExtractor(train_set)
	X, Y = fe.extract_feature()
	model.train(X, Y)

	validation_fe = FeatureExtractor(test_set)
	validation_X, validation_Y = validation_fe.extract_feature()
	Yp = model.predict(validation_X)
	rate = np.absolute(1 - np.divide(validation_Y, Yp))
	mape = np.mean(rate)
	difference = validation_Y - Yp
	validation_Y_mean = np.mean(validation_Y)
	validation_Y_std = np.std(validation_Y) 
	Yp_mean = np.mean(Yp)
	Yp_std = np.std(Yp)
	print 'validation_Y_mean'
	print 'validation_Y_std'
	print 'Yp_mean'
	print 'Yp_std'
	print validation_Y_mean
	print validation_Y_std
	print Yp_mean
	print Yp_std
	observe = np.concatenate((np.expand_dims(validation_Y, -1), np.expand_dims(Yp, -1), np.expand_dims(difference ,-1)), -1)
	print 'observe'
	print observe[0:30]
	
	print 'validation_Y'
	print validation_Y.shape
	print validation_Y[0:10]
	print 'Yp'
	print Yp.shape
	print Yp[0:10]
	print 'difference'
	print difference[0:100]
	
	print 'mape'
	print mape
	
	submit(model)

	#validation_X, _ = extract_feature()
	#Yp = model.predict(X)
	#evalutor = Evaluator()
	'''
def main_submit(args, p):
	import ipdb
	ipdb.set_trace()
	#args = get_args()
	#data_file = 'train_data'
	if args.task == 1:
		preprocess = travel_time_preprocess
		#data_file = 'train_data_for_training'
		data_file = 'train_data'
		validation_data_file = 'train_data_for_validation'

	else:
		preprocess = volume_preprocess
		data_file = 'task2_train_data_for_training'
		#data_file = 'task2_train_data'
		validation_data_file = 'task2_train_data_for_validation'

	train_set, _ = preprocess(data_file, rate=0.0)
	dev_set, _ = preprocess(validation_data_file, rate=0.0)
	fe = FeatureExtractor(train_set) if args.task == 1 else Task2FeatureExtractor(train_set)
	# X, Y = fe.extract_simple_feature()
	X, Y = fe.extract_feature()
	hyper_param_dict = {}
	hyper_param_dict['criterion'] = 'mae'
	hyper_param_dict['max_depth'] = 3
	hyper_param_dict['n_estimators'] = 40
	hyper_param_dict['min_samples_split'] = 3
	hyper_param_dict['min_weight_fraction_leaf'] = 0.2
	hyper_param_dict['min_samples_leaf'] = 3

	hyper_param_dict['hidden_layer_sizes'] = (100, 100, )
	hyper_param_dict['activation'] = 'relu'
	hyper_param_dict['solver'] = 'sgd'
	hyper_param_dict['alpha'] = 0.0001
	hyper_param_dict['learning_rate'] = 'adaptive'
	hyper_param_dict['max_iter'] = 200
	hyper_param_dict['early_stopping'] = True
	hyper_param_dict['beta_1'] = 0.9
	hyper_param_dict['beta_2'] = 0.999
	hyper_param_dict['epsilon'] = 1e-8
	model = Model(args, p)
	#model = Model(args, hyper_param_dict)
	model.train(X, Y)
	# evaluate model
	validation_fe = FeatureExtractor(dev_set) if args.task == 1 else Task2FeatureExtractor(dev_set)
	validation_X, validation_Y = validation_fe.extract_feature()
	s = model.score(validation_X, validation_Y)
	print s
	
	if args.task == 1:
		result_file = 'training_20min_avg_travel_time_8_with_model_%s' % args.model
		submit(model, result_file)
	else:
		result_file = 'training_20min_volumes_1_with_model_%s' % args.model
		volume_submit(model, result_file)

if __name__ == '__main__':
	args, p = main()
	main_submit(args, p)