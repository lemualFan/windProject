print("123")
import pandas as pd
from matplotlib import pyplot
data = pd.read_excel('data.xlsx')
data_names=['time','power_30s_avr','speed_wind_30s_avr','power_active','power_reactive','temp_de','speed_generator','temp_nde','speed_rotor','speed_high_shaft','temp_ambient','temp_main_bearing']
data.columns =  data_names

ndata = data[['time','power_30s_avr']]
#将时间转换成datetime格式
data['time'] = pd.to_datetime(data['time'],format="%Y-%m-%d %H:%M:%S")

#去除数据的object形式
data  = data.convert_objects(convert_numeric=True)
#去掉时间维度
data_droptime = data.drop(['time'],axis=1)
# data_len = len(data_names)
# i = 1
# for group in data_names:
#     pyplot.subplot(data_len, 1, i)
#     pyplot.plot(data[group])
#     pyplot.title(data.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()

pyplot.figure()
pyplot.subplot(2, 1, 1)
pyplot.plot(data['speed_wind_30s_avr'])
pyplot.title('speed_wind_30s_avr', y=0.5, loc='right')
pyplot.subplot(2, 1, 2)
pyplot.plot(data['power_30s_avr'])
pyplot.title('power_30s_avr', y=0.5, loc='right')


columns = ['power_30s_avr','speed_wind_30s_avr','power_active','power_reactive','temp_de','speed_generator','temp_nde','speed_rotor','speed_high_shaft','temp_ambient','temp_main_bearing']
pyplot.figure()
i = 1
for col in columns:
    pyplot.subplot(len(columns), 1, i)
    pyplot.plot(data[col])
    pyplot.title(col, y=0.3, loc='right')
    i+=1