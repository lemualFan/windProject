import pandas as pd
from matplotlib import pyplot

# data = pd.read_excel('data.xlsx')
# data_names = ['time', 'power_30s_avr', 'speed_wind_30s_avr', 'power_active', 'power_reactive', 'temp_de',
#               'speed_generator', 'temp_nde', 'speed_rotor', 'speed_high_shaft', 'temp_ambient', 'temp_main_bearing']
# data.columns = data_names


def load_data(data_path):
    data = pd.read_excel(data_path)
    data_names = ['time', 'power_30s_avr', 'speed_wind_30s_avr', 'power_active', 'power_reactive', 'temp_de',
                  'speed_generator', 'temp_nde', 'speed_rotor', 'speed_high_shaft', 'temp_ambient', 'temp_main_bearing']
    data.columns = data_names
    # 将时间转换成datetime格式
    data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d %H:%M:%S")
    # 去除数据的object形式
    data = data.convert_objects(convert_numeric=True)
    return data
