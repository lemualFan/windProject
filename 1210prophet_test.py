# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:25:54 2018

@author: flm
"""
import pandas as pd
import numpy as np
from fbprophet import Prophet

data = pd.read_excel('data.xlsx')
data_names=['time','power_30s_avr','speed_wind_30s_avr','power_active','power_reactive','temp_de','speed_generator','temp_nde','speed_rotor','speed_high_shaft','temp_ambient','temp_main_bearing']
data.columns =  data_names

ndata = data[['time','power_30s_avr']]

ndata.columns = ['ds','y']
#将时间转换成datetime形式
ndata['ds'] = pd.to_datetime(ndata['ds'])
#将时间设置为索引counter
#ndata = ndata.set_index('time',inplace=False,drop=True)

# Python
df = pd.read_csv('example_yosemite_temps.csv')
m = Prophet(changepoint_prior_scale=0.01).fit(df)
future = m.make_future_dataframe(periods=2, freq='H')
fcst = m.predict(future)
fig = m.plot(fcst)
