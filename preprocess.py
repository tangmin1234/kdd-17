from datetime import datetime, timedelta
import math
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize, normalize, Imputer, minmax_scale, scale

from utils import convert_to_interval_number, encode, type_of_date, one_hot_encode, piecewise_encode

file_suffix = '.csv'
path = '../data/dataSets/training/'

class DataSet(object):
	"""docstring for DataSet"""
	def __init__(self, data, data_type):
		self.data = data
		self.data_type = data_type
		self.data_keys = data.keys()
		self.num = len(data[self.data_keys[0]])

	def get_data(self):
		return self.data

	def get_mini_batch(self):
		pass

def read_data(trajectory_file, weather_file, path=path):
	trajectory_file_name = trajectory_file + file_suffix
	fr = open(path + trajectory_file_name, 'r')
	fr.readline()
	traj_data = fr.readlines()
	fr.close()

	print 'generate travel time dictory'
	travel_times = {}
	for i in tqdm(range(len(traj_data))):
		each_traj = traj_data[i].replace('"', '').split(',')
		intersection_id = each_traj[0]
		tollgate_id = each_traj[1]

		route_id = intersection_id + '-' + tollgate_id
		if route_id not in travel_times.keys():
			travel_times[route_id] = {}

		trace_start_time = each_traj[3]
		trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
		time_window_minute = int(math.floor(trace_start_time.minute / 20) * 20)
		start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
									trace_start_time.hour, time_window_minute, 0)

		tt = float(each_traj[-1])
		if start_time_window not in travel_times[route_id].keys():
			travel_times[route_id][start_time_window] = [tt]
		else:
			travel_times[route_id][start_time_window].append(tt)

	for route in travel_times.keys():
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		for time_window_start in route_time_windows:
			time_window_end = time_window_start + timedelta(minutes=20)
			tt_set = travel_times[route][time_window_start]
			avg_tt = round(sum(tt_set) / float(len(tt_set)))
			travel_times[route][time_window_start] = avg_tt

	weather_file_name = weather_file + file_suffix
	fr = open(path + weather_file_name)
	fr.readline()
	weather_data = fr.readlines()
	fr.close()

	print 'generate weather dictory'
	weather_stat = {}
	for i in tqdm(range(len(weather_data))):
		each_weather_data = weather_data[i].replace('"', '').split(',')
		date = each_weather_data[0]
		date = datetime.strptime(date, "%Y-%m-%d")
		time = datetime(date.year, date.month, date.day, int(each_weather_data[1]))
		statitics = [float(item) for item in each_weather_data[2:]]
		weather_stat[time] = statitics

	#test(travel_times)
	return travel_times, weather_stat

def test(travel_times):

	k = travel_times.keys()
	k.sort()
	print 
	for i in range(len(k)):
		route = k[i]
		print route
		print 'midautumn festival'
		d1 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 13]
		d2 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 14]
		d3 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 15]
		d4 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 16]
		d5 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 17]
		d6 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 18]
		d7 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 19]
		d8 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 20]

		print np.mean(np.array(d1))
		print np.mean(np.array(d2))
		print 'mf'
		print np.mean(np.array(d3))
		print np.mean(np.array(d4))
		print np.mean(np.array(d5))
		print np.mean(np.array(d6))
		print np.mean(np.array(d7))
		print np.mean(np.array(d8))

		print 'national day'
		d_1 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 29]
		d0 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==9 and date.day == 30]
		d1 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 1]
		d2 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 2]
		d3 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 3]
		d4 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 4]
		d5 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 5]
		d6 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 6]
		d7 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 7]
		d8 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 8]
		d9 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 9]
		d10 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 10]
		d11 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 11]
		d12 = [travel_times[route][date] for date in travel_times[route].keys() if date.month==10 and date.day == 12]
		print np.mean(np.array(d_1))
		print np.mean(np.array(d0))
		print 'nd'
		print np.mean(np.array(d1))
		print np.mean(np.array(d2))
		print np.mean(np.array(d3))
		print np.mean(np.array(d4))
		print np.mean(np.array(d5))
		print np.mean(np.array(d6))
		print np.mean(np.array(d7))
		print np.mean(np.array(d8))
		print np.mean(np.array(d9))
		print np.mean(np.array(d10))
		print np.mean(np.array(d12))

	return

def read_route_data(road_link_file, intersection2tollgate_file, path=path):
	road_link_file = road_link_file + file_suffix
	fr = open(path + road_link_file, 'r')
	fr.readline()
	road_links = fr.readlines()
	
	intersection2tollgate_file = intersection2tollgate_file + file_suffix
	fr = open(path + intersection2tollgate_file, 'r')
	fr.readline()
	routes = fr.readlines()

	road_links_dict = {}
	for i in range(len(road_links)):
		road_link = road_links[i].rstrip().replace('"', '').split(',')
		road_link = road_links[i].rstrip().split(',')
		road_link = [item.replace('"', '').split(',') for item in road_link]
		# print road_link
		road_link_id = road_link[0][0]
		if road_link_id not in road_links_dict:
			road_links_dict[road_link_id] = {}
			road_links_dict[road_link_id]['length'] = float(road_link[1][0])
			road_links_dict[road_link_id]['width'] = float(road_link[2][0])
			road_links_dict[road_link_id]['lanes'] = int(road_link[3][0])
			road_links_dict[road_link_id]['intop'] = len(road_link[4])
			road_links_dict[road_link_id]['outtop'] = len(road_link[5])
			road_links_dict[road_link_id]['lane_width'] = float(road_link[-1][0])

	route_info_dict = {}
	for i in range(len(routes)):
		route = routes[i].rstrip().replace('"', '').split(',')
		intersection_id = route[0]
		tollgate_id = route[1]
		route_id = intersection_id + '-' + tollgate_id
		if route_id not in route_info_dict:
			route_info_dict[route_id] = route[2:]

	return road_links_dict, route_info_dict

def construct_data(trajectory_file=None, weather_file=None, road_link_file=None, intersection2tollgate_file=None, data_file=None):
	travel_times, weather_stat = read_data(trajectory_file, weather_file)
	road_links_dict, route_info_dict = read_route_data(road_link_file, intersection2tollgate_file)

	train_data = {}
	interval = []
	moment = []
	weather = []
	date_type = []
	route_info = []
	object_weather = []
	consecutive_avg_tt = []
	object_consecutive_avg_tt = []

	#import ipdb
	#ipdb.set_trace()

	for route in tqdm(travel_times.keys()):
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		for time_window_start in tqdm(route_time_windows):
			if time_window_start + timedelta(hours=4) in route_time_windows:
			#and 6 <= time_window_start.hour <= 18:
			#and (time_window_start.hour == 6 or time_window_start.hour == 15):
				c_a_t = []
				for i in range(6):
					current_time_start = time_window_start + timedelta(minutes=20*i)
					if current_time_start in route_time_windows:
						c_a_t.append(travel_times[route][current_time_start])
					else:
						c_a_t.append(np.NaN)
				o_c_a_t = []
				for i in range(6):
					current_time_start = time_window_start + timedelta(hours=2, minutes=20*i)
					if current_time_start in route_time_windows:
						o_c_a_t.append(travel_times[route][current_time_start])	
					else:
						o_c_a_t.append(np.NaN)

				year, month, day, hour = time_window_start.year, time_window_start.month, time_window_start.day, time_window_start.hour
				date1 = datetime(year, month, day, int(((hour+1)//3)*3)%24)
				date2 = datetime(year, month, day, int(((hour+3)//3)*3)%24)
				assert date1.hour % 3 == 0
				assert date2.hour % 3 == 0
				if date1 not in weather_stat.keys() or date2 not in weather_stat.keys() :
					continue
				#weather_item = weather_stat[date1] if weather_stat[date1][2] >= 360 else weather_stat[date1 - timedelta(hours=3)] or weather_stat[date1 + timedelta(hours=3)]
				#object_weather_item = weather_stat[date2] if weather_stat[date2][2] >= 360 else weather_stat[date2 - timedelta(hours=3) or weather_stat[date2 + timedelta(hours=3)]]
				weather_item = weather_stat[date1]
				if weather_item[2] >= 360:
					weather_item[2] = np.NaN
				object_weather_item = weather_stat[date2]
				if object_weather_item[2] >= 360:
					object_weather_item[2] = np.NaN
				assert len(o_c_a_t) == 6
				if type_of_date(datetime(time_window_start.year, time_window_start.month, time_window_start.day)) == 3:
					continue
				if (time_window_start.month == 9 and time_window_start.day == 14) or (time_window_start.month==9 and time_window_start.day==30):
					continue
				for i, o_c_a_t_item in enumerate(o_c_a_t):
					consecutive_avg_tt.append(c_a_t)
					interval.append(one_hot_encode(i, bit_num=6))
					object_consecutive_avg_tt.append([o_c_a_t_item])
					route_info.append(get_route_info(route, road_links_dict, route_info_dict))
					weather.append(weather_item)
					object_weather.append(object_weather_item)
					moment.append(piecewise_encode(time_window_start+timedelta(hours=2), bit_num=6))
					date_type.append(encode(type_of_date(datetime(time_window_start.year, time_window_start.month, time_window_start.day)), bit_num=2))


	train_data['route_info'] = route_info
	train_data['weather'] = weather
	train_data['object_weather'] = object_weather
	train_data['consecutive_avg_tt'] = consecutive_avg_tt
	train_data['object_consecutive_avg_tt'] = object_consecutive_avg_tt
	train_data['moment'] = moment
	train_data['interval'] = interval
	train_data['date_type'] = date_type

	suffix = '.json'
	data_file = data_file + suffix
	json.dump(train_data, open(path + 'src2_data/' + data_file, 'w'))
	#test_data_file = 'test_data' + suffix
	#json.dump(train_data, open(path + 'src2_data/' + test_data_file, 'w'))


def preprocess(data_file=None, rate=0.1, path=path, shuffle=True):
	suffix = '.json'
	train_data_file_name =  data_file+ suffix
	print 'read data from %s' % train_data_file_name
	with open(path + 'src2_data/' + train_data_file_name, 'r') as fh:
			train_data = json.load(fh)

	data_keys = train_data.keys()
	for key in data_keys:
		if key.endswith('avg_tt'):
			imputer = Imputer()
			imputer.fit(np.array(train_data[key]))
			train_data[key] = imputer.transform(train_data[key])
		#if key.endswith('moment') or key.endswith('interval'):
		#	train_data[key] = minmax_scale(np.array(train_data[key]))

		if key.endswith('weather'):
			imputer = Imputer()
			imputer.fit(np.array(train_data[key]))
			train_data[key] = imputer.transform(train_data[key])

	#data = {key: scale(np.array(item), axis=0) for key, item in train_data.items() \
	data = {key: item for key, item in train_data.items() \
												if not key.endswith('moment') \
												and not key.endswith('object_consecutive_avg_tt') \
												and not key.endswith('interval')\
												and not key.endswith('date_type') }
	
	for key in data_keys:
		#if not key.endswith('moment') and not key.endswith('object_consecutive_avg_tt') and not key.endswith('interval'):
		#	print data[key].mean(axis=0)
		#	print data[key].std(axis=0)
		#print key, data[]
		pass		
	data['moment'] = train_data['moment']
	data['interval'] = train_data['interval']
	data['date_type'] = train_data['date_type']
	data['object_consecutive_avg_tt'] = np.squeeze(train_data['object_consecutive_avg_tt'])
	#data['moment'] = np.expand_dims(train_data['moment'], axis=-1)
	#data['interval'] = np.expand_dims(train_data['interval'], axis=-1)
	#data['object_consecutive_avg_tt'] = np.expand_dims(train_data['object_consecutive_avg_tt'], axis=-1)
	
	train_set, dev_set = split_data(data, rate=rate, shuffle=shuffle)
	return train_set, dev_set

def split_data(data, rate=0.1, shuffle=True):
	train_data = {}
	dev_data = {}
	assert type(data) == dict
	data_keys = data.keys()
	train_num = len(data[data_keys[0]])
	train_set_num = int(train_num * (1.0 - rate))
	dev_set_num = train_num - train_set_num
	valid_idx = range(train_num)
	if shuffle:
		np.random.shuffle(valid_idx)
	shuffle_idx = valid_idx
	train_set_idx = shuffle_idx[:train_set_num]
	dev_set_idx = shuffle_idx[train_set_num:]

	for key in data_keys:
		#print key
		train_data[key] = [data[key][idx] for idx in train_set_idx]
		dev_data[key] = [data[key][idx] for idx in dev_set_idx]

	train_set = DataSet(train_data, 'train')
	dev_set = DataSet(dev_data, 'dev')
	return train_set, dev_set

def get_route_info(route, road_links_dict, route_info_dict):
	route_info = []
	route_length = 0
	route_width = 0
	route_lanes = 0
	route_intops = 0
	route_outtops = 0
	route_links = route_info_dict[route]
	for link_id in route_links:
		#route_length += road_links_dict[link_id]['length'] #  * road_links_dict[link_id]['width']
		route_length += road_links_dict[link_id]['length']
		route_width += road_links_dict[link_id]['width']
		route_intops += road_links_dict[link_id]['intop']
		route_outtops += road_links_dict[link_id]['outtop']
		route_lanes = road_links_dict[link_id]['lanes']
	route_info.append(route_length)
	#route_info.append(route_lanes)
	#route_info.append(route_width/len(route_links))
	route_info.append(route_intops)
	route_info.append(route_outtops)
	return route_info

if __name__ == '__main__':

	import ipdb
	ipdb.set_trace()
	#trajectory_file = 'trajectories(table 5)_training_for_training'
	#trajectory_file = 'trajectories(table 5)_training_for_validation'
	trajectory_file = 'trajectories(table 5)_training'

	weather_file = 'weather (table 7)_training_update'

	road_link_file = 'links(table 3)'
	intersection2tollgate_file = 'routes(table 4)'

	#data_file = 'test_data'
	#data_file = 'partial(6_18)_train_data_for_training'
	#data_file = 'train_data_for_validation'
	#data_file = 'train_data_for_validation_no_use'
	#data_file = 'partial_train_data_dummy_variable'
	data_file = 'train_data'
	construct_data(trajectory_file, weather_file, road_link_file, intersection2tollgate_file, data_file)
	
	train_set, dev_set = preprocess(data_file)