from datetime import datetime, timedelta
import numpy as np
import json
from tqdm import tqdm
from preprocess import read_data, read_route_data, get_route_info, preprocess
from volume_preprocess import volume_preprocess, read_volume_data, get_tollgate_direction_info
from utils import encode, convert_to_interval_number, convert_to_time_window, type_of_date, one_hot_encode, piecewise_encode, type_of_vehicle
from FeatureExtractor import FeatureExtractor, Task2FeatureExtractor

path_test = '../data/dataSets/testing_phase1/'
suffix = '.csv'
path = '../data/dataSets/training/'


def construct_data(test_trajectory_file, test_weather_file, road_link_file, intersection2tollgate_file, data_file):
	travel_times, weather_stat = read_data(test_trajectory_file, test_weather_file, path=path_test)
	road_links_dict, route_info_dict = read_route_data(road_link_file, intersection2tollgate_file, path=path)

	route_id = []
	time = []
	test_data = {}
	date_type = []
	interval = []
	moment = []
	weather = []
	route_info = []
	object_weather = []
	consecutive_avg_tt = []
	object_consecutive_avg_tt = []

	routes = list(travel_times.keys())
	routes.sort()
	for i in range(len(routes)):
		route = routes[i]
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		#partial_route_time_windows = route_time_windows[::6]
		partial_route_time_windows = []
		date = datetime(2016, 10, 18)
		while date < datetime(2016, 10, 25):
			date = date + timedelta(hours=6)
			partial_route_time_windows.append(date)
			date = date + timedelta(hours=9)
			partial_route_time_windows.append(date)
			date = datetime(date.year, date.month, int(date.day+1))

		for i in range(len(partial_route_time_windows)):
			current_time_window = partial_route_time_windows[i]
			c_a_t = []
			for i in range(6):
				if current_time_window + timedelta(minutes=20*i) in route_time_windows:
					c_a_t.append(travel_times[route][current_time_window + timedelta(minutes=20*i)])
				else:
					c_a_t.append(np.NaN)

			time_window_start = current_time_window + timedelta(hours=1)
			year, month, day, hour = time_window_start.year, time_window_start.month, time_window_start.day, time_window_start.hour
			date1 = datetime(year, month, day, int(((hour+1)//3)*3)%24)
			date2 = datetime(year, month, day, int(((hour+3)//3)*3)%24)
			assert date1.hour % 3 == 0
			assert date2.hour % 3 == 0
			weather_item = weather_stat[date1]
			if weather_item[2] >= 360:
				weather_item[2] = np.NaN
			object_weather_item = weather_stat[date2]
			if object_weather_item[2] >= 360:
				object_weather_item[2] = np.NaN
			
			for i in range(6):
				consecutive_avg_tt.append(c_a_t)
				interval.append(one_hot_encode(i, bit_num=6))
				object_consecutive_avg_tt.append([np.NaN])
				route_info.append(get_route_info(route, road_links_dict, route_info_dict))
				weather.append(weather_item)
				object_weather.append(object_weather_item)
				moment.append(piecewise_encode(current_time_window+timedelta(hours=2), bit_num=6))
				route_id.append(route)
				time.append(current_time_window)
				date_type.append(encode(type_of_date(datetime(time_window_start.year, time_window_start.month, time_window_start.day)), bit_num=2))

	test_data['route_info'] = route_info
	test_data['weather'] = weather
	test_data['object_weather'] = object_weather
	test_data['consecutive_avg_tt'] = consecutive_avg_tt
	test_data['object_consecutive_avg_tt'] = object_consecutive_avg_tt
	test_data['moment'] = moment
	test_data['interval'] = interval
	test_data['date_type'] = date_type

	suffix = '.json'
	test_data_file = 'test_data' + suffix
	json.dump(test_data, open(path_test + 'src2_data/' + test_data_file, 'w'))
	return route_id, time

def write_into_file(route_id, time, test_set, predicted_Y, result_file):
	#import ipdb
	#ipdb.set_trace()
	data = test_set.data
	i = 0
	result_file_name = result_file + suffix
	with open(path_test + 'result/' + result_file_name, 'w') as fw:
		fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
		for i in range(test_set.num):
			route = route_id[i]
			time_window_start, time_window_end = convert_to_time_window(time[i], data['interval'][i])
			avg_tt = predicted_Y[i]
			out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
									'"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
									'"' + str(avg_tt) + '"']) + '\n'
			i += 1
	   		fw.writelines(out_line)

def submit(model, result_file):
	#import ipdb
	#ipdb.set_trace()
	print 'generate result stored in %s' % (result_file + suffix)
	test_weather_file = 'weather (table 7)_test1'
	test_trajectory_file = 'trajectories(table 5)_test1'
	road_link_file = 'links(table 3)'
	intersection2tollgate_file = 'routes(table 4)'
	data_file = 'test_data'
	route_id, time = construct_data(test_trajectory_file, test_weather_file, road_link_file, intersection2tollgate_file, data_file)
	test_set, _ = preprocess(data_file, 0.0, path_test, shuffle=False)
	test_set_ = test_set
	fe = FeatureExtractor(test_set_)
	test_X, test_Y = fe.extract_feature()
	Yp = model.predict(test_X)
	write_into_file(route_id, time, test_set, Yp, result_file)

def construct_volume_data(volume_file, weather_file, road_link_file, intersection2tollgate_file, data_file):

	volumes, weather_stat = read_volume_data(volume_file, weather_file, path=path_test)
	road_links_dict, route_info_dict = read_route_data(road_link_file, intersection2tollgate_file)
	
	tollgate_direction_id = []
	time = []
	vt_num = []
	test_data = {}
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
	tollgate_directions.sort()
	for idx in tqdm(range(len(tollgate_directions))):
		time_windows = list(volumes[tollgate_directions[idx]].keys())
		time_windows.sort()
		for time_window_start in time_windows:
			if time_window_start.hour == 6 and time_window_start.minute == 0 or time_window_start.hour == 15 and time_window_start.minute==0:
				c_v = []
				e_t_c = []
				for i in range(6):
					current_time_start = time_window_start + timedelta(minutes=20*i)
					if current_time_start in time_windows:
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
					tollgate_direction_id.append(tollgate_directions[idx])
					time.append(time_window_start)

	#test_data['vehicle_type'] = vehicle_type
	test_data['consecutive_vol'] = consecutive_vol
	test_data['object_consecutive_vol'] = object_consecutive_vol
	test_data['weather'] = weather
	test_data['object_weather'] = object_weather
	test_data['interval'] = interval
	test_data['moment'] = moment
	test_data['date_type'] = date_type
	test_data['etc'] = etc
	#test_data['object_etc'] = object_etc
	test_data['tollgate_direction_info'] = tollgate_direction_info

	suffix = '.json'
	data_file = data_file + suffix
	json.dump(test_data, open(path_test + 'src2_data/' + data_file, 'w'))

	return tollgate_direction_id, time

def task2_write_into_file(tollgate_direction_id, time, test_set, Yp, result_file):

	data = test_set.data
	i = 0
	result_file_name = result_file + suffix
	with open(path_test + 'result/' + result_file_name, 'w') as fw:
		fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')

		for i in range(5*7*12):
			tollgate_id = tollgate_direction_id[i].replace("'", '')[0]
			direction = tollgate_direction_id[i].replace("'", '')[-1]
			time_window_start, time_window_end = convert_to_time_window(time[i], data['interval'][i])
			vol = Yp[i]

			out_line = ','.join(['"' + str(tollgate_id) + '"', 
								'"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
								'"' + str(direction) + '"',
								'"' + str(vol) + '"',
								]) + '\n'
			fw.writelines(out_line)

def volume_submit(model, result_file):
	print 'generate result stored in %s' % (result_file + suffix)
	test_weather_file = 'weather (table 7)_test1'
	test_volume_file = 'volume(table 6)_test1'
	road_link_file = 'links(table 3)'
	intersection2tollgate_file = 'routes(table 4)'
	data_file = 'task2_test_data'
	tollgate_direction_id, time = construct_volume_data(test_volume_file, test_weather_file, road_link_file, intersection2tollgate_file, data_file)
	test_set, _ = volume_preprocess(data_file, 0.0, path_test, shuffle=False)
	test_set_ = test_set
	fe = Task2FeatureExtractor(test_set_)
	test_X, test_Y = fe.extract_feature()
	Yp = model.predict(test_X)
	task2_write_into_file(tollgate_direction_id, time, test_set, Yp, result_file)

if __name__ == '__main__':
	model = None
	result_file = 'test_result_file'
	#submit(model, result_file)
	volume_submit(model, result_file)
