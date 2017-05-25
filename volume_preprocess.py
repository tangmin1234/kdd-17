import numpy as np
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import math
from sklearn.preprocessing import Imputer

from preprocess import read_route_data, DataSet, split_data
from utils import one_hot_encode, piecewise_encode, encode, type_of_date, type_of_vehicle

file_suffix = '.csv'
path = '../data/dataSets/training/'


def read_volume_data(volume_file, weather_file, path=path):
	volume_file_name = volume_file + file_suffix
	fr = open(path + volume_file_name,'r')
	fr.readline()
	vol_data = fr.readlines()
	fr.close()

	print 'generate volume dictionary'
	volumes = {}
	for i in tqdm(range(len(vol_data))):
		each_pass = vol_data[i].replace('"', '').split(',')
		tollgate_id = each_pass[1]
		direction = each_pass[2]
		tollgate_direction = tollgate_id + '-' + direction
		has_etc = each_pass[4]
		vehicle_model = int(each_pass[3])
		vehicle_type = each_pass[5]
		if vehicle_type == '\n':
			vehicle_type = 's'
		pass_time = each_pass[0]

		pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
		time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
		#print pass_time
		start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)


		if tollgate_direction not in volumes:
			volumes[tollgate_direction] = {}

		#if vehicle_type not in volumes[tollgate_direction]:
		#	volumes[tollgate_direction][vehicle_type] = {}

		if start_time_window not in volumes[tollgate_direction]:
			volumes[tollgate_direction][start_time_window] = {}

		if has_etc not in volumes[tollgate_direction][start_time_window]:
			volumes[tollgate_direction][start_time_window][has_etc] = [0] * 8

		volumes[tollgate_direction][start_time_window][has_etc][vehicle_model] += 1

		'''if start_time_window not in volumes:
			volumes[start_time_window] = {}
		if tollgate_id not in volumes[start_time_window]:
			volumes[start_time_window][tollgate_id] = {}
		if direction not in volumes[start_time_window][tollgate_id]:
			volumes[start_time_window][tollgate_direction] = {}
		if vehicle_type not in volumes[start_time_window][tollgate_direction]:
			volumes[start_time_window][tollgate_direction][vehicle_type] = {}
		if has_etc not in volumes[start_time_window][tollgate_direction][vehicle_type]:
			volumes[start_time_window][tollgate_direction][vehicle_type][has_etc] = [0] * 8 # 8 is nubmer of vehicle model
		volumes[start_time_window][tollgate_direction][vehicle_type][has_etc][vehicle_model] += 1
		'''
	weather_file_name = weather_file + file_suffix
	fr = open(path + weather_file_name)
	fr.readline()
	weather_data = fr.readlines()
	fr.close()

	print 'generate weather dictionary'
	weather_stat = {}
	for i in tqdm(range(len(weather_data))):
		each_weather_data = weather_data[i].replace('"', '').split(',')
		date = each_weather_data[0]
		date = datetime.strptime(date, "%Y-%m-%d")
		time = datetime(date.year, date.month, date.day, int(each_weather_data[1]))
		statitics = [float(item) for item in each_weather_data[2:]]
		weather_stat[time] = statitics


	return volumes, weather_stat

def construct_data(volume_file=None, weather_file=None, road_link_file=None, intersection2tollgate_file=None, data_file=None):
	volumes, weather_stat = read_volume_data(volume_file, weather_file)
	road_links_dict, route_info_dict = read_route_data(road_link_file, intersection2tollgate_file)
	
	train_data = {}
	vehicle_type = []
	tollgate_direction_info = []
	etc = []
	object_etc = []
	interval = []
	moment = []
	weather = []
	date_type = []
	object_weather = []
	consecutive_vol = []
	object_consecutive_vol = []

	
	tollgate_directions = volumes.keys()
	for idx in tqdm(range(len(tollgate_directions))):
		#vts = list(volumes[tollgate_directions[idx]].keys())
		#for vt in vts:
		time_windows = list(volumes[tollgate_directions[idx]].keys())
		time_windows.sort()
		for time_window_start in tqdm(time_windows):
			if time_window_start + timedelta(hours=4) in time_windows \
				and time_window_start.hour == 6 or time_window_start.hour == 15:
				#and time_window_start + timedelta(hours=4) < datetime(2016, 10, 11):
				c_v = []
				e_t_c = []
				for i in range(6):
					current_time_start = time_window_start + timedelta(minutes=20*i)
					if current_time_start in time_windows:
						# etc is list
						#e_t_c = reduce(lambda x, y: x + y, [volumes[tollgate_directions[idx]][current_time_start]['0'], volumes[tollgate_directions[idx]][current_time_start]['1']]) 
						volumes[tollgate_directions[idx]][current_time_start]['0'] = volumes[tollgate_directions[idx]][current_time_start]['0'] \
																						if '0' in volumes[tollgate_directions[idx]][current_time_start].keys() else [0] * 8
						volumes[tollgate_directions[idx]][current_time_start]['1'] = volumes[tollgate_directions[idx]][current_time_start]['1'] \
																						if '1' in volumes[tollgate_directions[idx]][current_time_start].keys() else [0] * 8


						e_t_c = volumes[tollgate_directions[idx]][current_time_start]['0'] + volumes[tollgate_directions[idx]][current_time_start]['1']
						c_v.append(sum(e_t_c))

					else:
						e_t_c = [np.NaN] * 16
						c_v.append(np.NaN)

				o_c_v = []
				o_e_t_c = []
				for i in range(6):
					current_time_start = time_window_start + timedelta(hours=2, minutes=20*i)
					if current_time_start in time_windows:
						volumes[tollgate_directions[idx]][current_time_start]['0'] = volumes[tollgate_directions[idx]][current_time_start]['0'] \
																						if '0' in volumes[tollgate_directions[idx]][current_time_start].keys() else [0] * 8
						volumes[tollgate_directions[idx]][current_time_start]['1'] = volumes[tollgate_directions[idx]][current_time_start]['1'] \
																						if '1' in volumes[tollgate_directions[idx]][current_time_start].keys() else [0] * 8
						o_e_t_c = volumes[tollgate_directions[idx]][current_time_start]['0'] + volumes[tollgate_directions[idx]][current_time_start]['1']
						o_e_t_c = reduce(lambda x, y: x + y, [volumes[tollgate_directions[idx]][current_time_start]['0'], volumes[tollgate_directions[idx]][current_time_start]['1']]) 
						o_c_v.append(sum(o_e_t_c))	
					
					else:
						o_e_t_c = [np.NaN] * 16
						o_c_v.append(np.NaN)

				
				year, month, day, hour = time_window_start.year, time_window_start.month, time_window_start.day, time_window_start.hour
				date1 = datetime(year, month, day, int(((hour+1)//3)*3)%24)
				date2 = datetime(year, month, day, int(((hour+3)//3)*3)%24)
				assert date1.hour % 3 == 0
				assert date2.hour % 3 == 0
				if date1 not in weather_stat.keys() or date2 not in weather_stat.keys() :
					continue

				weather_item = weather_stat[date1]
				if weather_item[2] >= 360:
					weather_item[2] = np.NaN
				object_weather_item = weather_stat[date2]
				if object_weather_item[2] >= 360:
					object_weather_item[2] = np.NaN
				assert len(o_c_v) == 6
				if type_of_date(datetime(time_window_start.year, time_window_start.month, time_window_start.day)) == 3:
					continue
				if (time_window_start.month == 9 and time_window_start.day == 14) or (time_window_start.month==9 and time_window_start.day==30):
					continue

				for i, o_c_v_item in enumerate(o_c_v):
					#vehicle_type.append(one_hot_encode(type_of_vehicle(vt), bit_num=3))
					consecutive_vol.append(c_v)
					object_consecutive_vol.append([o_c_v_item])
					weather.append(weather_item)
					object_weather.append(object_weather_item)
					interval.append(one_hot_encode(i, bit_num=6))
					moment.append(piecewise_encode(time_window_start+timedelta(hours=2), bit_num=6))
					date_type.append(encode(type_of_date(datetime(time_window_start.year, time_window_start.month, time_window_start.day)), bit_num=2))
					etc.append(e_t_c)
					#object_etc.append(o_e_t_c)
					tollgate_direction_info.append(get_tollgate_direction_info(tollgate_directions[idx], road_links_dict, route_info_dict))


	#train_data['vehicle_type'] = vehicle_type
	train_data['consecutive_vol'] = consecutive_vol
	train_data['object_consecutive_vol'] = object_consecutive_vol
	train_data['weather'] = weather
	train_data['object_weather'] = object_weather
	train_data['interval'] = interval
	train_data['moment'] = moment
	train_data['date_type'] = date_type
	train_data['etc'] = etc
	#train_data['object_etc'] = object_etc
	train_data['tollgate_direction_info'] = tollgate_direction_info

	suffix = '.json'
	data_file = data_file + suffix
	json.dump(train_data, open(path + 'src2_data/' + data_file, 'w'))


def get_tollgate_direction_info(tollgate_direction, road_links_dict, route_info_dict):
	tollgate_direction_info = []
	#tollgate_id = '-'.split(tollgate_direction)[0]
	tollgate_id = tollgate_direction.replace("'", '')[0]
	link = []
	for key in route_info_dict.keys():
		if key.endswith(tollgate_id):
			link = route_info_dict[key][-1]
			#link.append(route_info_dict[key][-1])
			break
	lanes = road_links_dict[link]['lanes']
	tollgate_direction_info.append(lanes)
	return tollgate_direction_info

def volume_preprocess(data_file, rate=0.1, path=path, shuffle=True):
	suffix = '.json'
	data_file_name =  data_file+ suffix
	print 'read data from %s' % data_file_name
	with open(path + 'src2_data/' + data_file_name, 'r') as fh:
			train_data = json.load(fh)

	data_keys = train_data.keys()
	for key in data_keys:
		if key.endswith('vol') or key.endswith('etc'):
			imputer = Imputer()
			imputer.fit(np.array(train_data[key]))
			train_data[key] = imputer.transform(np.array(train_data[key]))
		#if key.endswith('moment') or key.endswith('interval'):
		#	train_data[key] = minmax_scale(np.array(train_data[key]))

		if key.endswith('weather'):
			imputer = Imputer()
			imputer.fit(np.array(train_data[key]))
			train_data[key] = imputer.transform(np.array(train_data[key]))

	data = {key: item for key, item in train_data.items() \
												if not key.endswith('moment') \
												and not key.endswith('object_consecutive_vol') \
												and not key.endswith('interval')\
												and not key.endswith('date_type')}
	
	for key in data_keys:
		if not key.endswith('moment') and not key.endswith('object_consecutive_avg_tt') and not key.endswith('interval'):
		#	print data[key].mean(axis=0)
		#	print data[key].std(axis=0)
		#print key, data[]
			pass		
	data['moment'] = train_data['moment']
	data['interval'] = train_data['interval']
	data['date_type'] = train_data['date_type']
	#data['vehicle_type'] = train_data['vehicle_type']
	data['object_consecutive_vol'] = np.squeeze(train_data['object_consecutive_vol'])

	train_set, dev_set = split_data(data, rate=rate, shuffle=shuffle)
	return train_set, dev_set

if __name__ == '__main__':
	import ipdb
	ipdb.set_trace()
	weather_file = 'weather (table 7)_training_update'
	road_link_file = 'links(table 3)'
	intersection2tollgate_file = 'routes(table 4)'	
	
	volume_file = 'volume(table 6)_training'
	#volume_file = 'volume(table 6)_training_for_training'
	validation_volume_file = 'volume(table 6)_training_for_validation'
	
	data_file = 'task2_train_data'
	#data_file = 'task2_train_data_for_training'
	validation_data_file = 'task2_train_data_for_validation'
	construct_data(volume_file, weather_file, road_link_file, intersection2tollgate_file, data_file)
	#construct_data(validation_volume_file, weather_file, road_link_file, intersection2tollgate_file, validation_data_file)
	train_set, dev_set = volume_preprocess(data_file, 0.0)