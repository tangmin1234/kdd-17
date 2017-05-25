from tqdm import tqdm

from FeatureExtractor import FeatureExtractor, Task2FeatureExtractor
from regressor import Model
from utils import combine_parameter, pprint


class HyperParameterTunning(object):
	"""docstring for HyperParameterTunning"""
	def __init__(self, args):
		self.args = args
		self.build_hyper_parameter_dict()

	def build_hyper_parameter_dict(self):
		model_hyper_parameter_dict = {}
		model_hyper_parameter_dict['lr'] = {}
		model_hyper_parameter_dict['ridge'] = {}
		model_hyper_parameter_dict['lasso'] = {}
		model_hyper_parameter_dict['knn'] = {'n_neighbors': [7], \
											'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], \
											'leaf_size': [15, 20, 25, 30], \
											'p': [1, 2, 3]}
	
		model_hyper_parameter_dict['rf'] = {'n_estimators': [25, 30, 40], \
											'criterion': ['mae'], \
											'min_samples_split': [3, 4, 5, 6], \
											'min_samles_leaf': [2, 3], \
											'min_weight_fraction_leaf': [0.0, 0.15, 0.1, 0.2, 0.25]}
		model_hyper_parameter_dict['dt'] = {}
		model_hyper_parameter_dict['gb'] = {'n_estimators': [10, 20, 30], \
											'criterion': ['mae'], \
											'max_depth': [2, 3, 4, 5], \
											'min_samples_split': [2, 3, 4, 5], \
											'min_samples_leaf': [1, 2], \
											'min_weight_fraction_leaf': [0.0, 0.15, 0.1, 0.2, 0.25]}
		model_hyper_parameter_dict['nn'] = {'hidden_layer_sizes': [], \
											'': [], \
											'': [], \
											'': [], \
											'': [], \
											'': [], \
											'': []}

		self.model_hyper_parameter_dict = model_hyper_parameter_dict

	def tune_parameter(self, dataset):
		args = self.args
		hyperparameter_dict = self.model_hyper_parameter_dict[args.model]
		
		#choice = {key: len(value) for key, value in hyperparameter_dict.items()}
		choices = combine_parameter(hyperparameter_dict)

		print '\n%s model parameter tunning ...\n' % '{}'.format(args.model)
		best_score = 1e8
		best_choice = None
		for i in tqdm(range(len(choices))):
			hyperparameter = choices[i]
			# print hyperparameter
			model = Model(args, hyperparameter)
			fe = FeatureExtractor(dataset[0]) if args.task == 1 else Task2FeatureExtractor(dataset[0])
			X, Y = fe.extract_feature()
			model.train(X, Y)
			valid_fe = FeatureExtractor(dataset[1]) if args.task == 1 else Task2FeatureExtractor(dataset[1])
			validation_X, validation_Y = valid_fe.extract_feature()
			score = model.score(validation_X, validation_Y)
			pprint(hyperparameter)
			print 'score %f' % score
			print '\n\n\n'
			if score < best_score:
				best_score = score
				best_choice = choices[i]
		self.best_score = best_score
		self.best_paramater = best_choice

	def get_best_pamater(self):
		print 'Note:this parameter is the best parameter for %s model' % '{}'.format(self.args.model)
		return self.best_paramater, self.best_score
