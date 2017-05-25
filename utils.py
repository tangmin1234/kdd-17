import numpy as np
from datetime import timedelta, datetime

def pprint(dictionay):
	keys = dictionay.keys()
	keys.sort()
	for i in range(len(keys)):
		print keys[i], dictionay[keys[i]]

def convert_to_interval_number(datetime):
	return datetime.hour * 3 + datetime.minute / 20

def convert_to_time_window(time, interval):
	#interval = interval.tolist()
	#interval = interval[::-1]
	assert isinstance(interval, list)
	assert len(interval) == 6
	start = time + timedelta(hours=2) + timedelta(minutes=20*(decode(interval)))
	end = start + timedelta(minutes=20)
	return start, end

def piecewise_encode(date, bit_num=4):
	value = None
	if 0 <= date.hour < 4:
		value = 0
	elif 4 <= date.hour < 8:
		value = 1
	elif 8 <= date.hour < 12:
		value = 2		
	elif 12 <= date.hour < 16:
		value = 3
	elif 16 <= date.hour < 20:
		value = 4
	else:
		value = 5
	return one_hot_encode(value, bit_num=bit_num)

def partial(datetime):
	if (datetime.hour > 6 and datetime.hour < 11) or (datetime.hour > 14 and datetime.hour < 19):
		return True
	return False

def encode(number, bit_num=8):
	code = []
	x = number % 2
	i = 0
	while i < bit_num:
		code.append(x)
		number = number // 2
		x = number % 2
		i = i + 1
	return code

def one_hot_encode(number, bit_num=6):
	code = [0] * bit_num
	for i in range(bit_num):
		if i == number:
			code[i] = 1
	return code
	
def decode(bits, bit_num=6):
	for i in range(bit_num):
		if bits[i] == 1:
			return i
	raise ValueError('invalid bits')
	#number = 0
	#for i in range(bit_num):
	#	number += int(bits[i])* 2**i
	#assert number < 8

def get_holiday_list(start, end):
	calendar = {}
	calendar['workday'] = []
	calendar['weekend'] = []
	calendar['holiday'] = []
	current = datetime(2016, 9, 15)
	for i in range(3):
		calendar['holiday'].append(current + timedelta(days=i))
	current = datetime(2016, 10, 1)
	for i in range(7):
		calendar['holiday'].append(current + timedelta(days=i))

	i = 1
	current = start
	while current <= end:
		if current in calendar['holiday']:
			pass
		else:
			if i == 5 or i == 6:
				calendar['weekend'].append(current)
			else:
				calendar['workday'].append(current)
		i = (i + 1) % 7
		current  = current + timedelta(days=1)
	# modifite calendar
	date1 = datetime(2016, 10, 8)
	date2 = datetime(2016, 10, 9)
	date3 = datetime(2016, 9, 18)
	calendar['weekend'].remove(date1)
	calendar['weekend'].remove(date2)
	calendar['weekend'].remove(date3)
	calendar['workday'].append(date1)
	calendar['workday'].append(date2)
	calendar['workday'].append(date3)
	return calendar

holiday_list = get_holiday_list(datetime(2016,7,19), datetime(2016, 10, 24))

def type_of_vehicle(vt):
	if vt == 's':
		return 1
	elif vt == '0\n':
		return 2
	elif vt == '1\n':
		return 3
	else:
		raise ValueError('invalid vehicle type')
def type_of_date(date):
	if date in holiday_list['holiday']:
		value =  3
	elif date in holiday_list['weekend']:
		value = 2
	elif date in holiday_list['workday']:
		value = 1
	else:
		raise ValueError('invalid date')
	return value


def combine_parameter(dictionay):
	keys = dictionay.keys()
	keys.sort()
	#lists = [l for k, l in dictionay.items()]
	lists = [dictionay[k] for k in keys]
	total = reduce(lambda x, y: x * y, map(len, lists)) if len(lists) > 1 else len(lists[0])
	ret_list = []
	for j in range(total):
		step = total
		temp_item = {}
		for i, key in enumerate(keys):
			l = lists[i]
			step /= len(l)
			temp_item[key] = l[j / step % len(l)]
		ret_list.append(temp_item)

	return ret_list

if __name__ == '__main__':
	d = {}
	d['x'] = [1, 2, 3]
	d['y'] = ['a', 'b', 'c', 'd']
	d['z'] = ['M', 'N']
	ret = combine_parameter(d)
	print ret