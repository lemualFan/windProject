from  data import loaddata
from config import RAW_DATA,PROCESS_LEVEL1
import pandas as pd
from series_to_supervised_learning import series_to_supervised
data = loaddata.load_data(RAW_DATA)
data.to_csv(PROCESS_LEVEL1)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error


ndata = data[['time','power_30s_avr','speed_wind_30s_avr']]
values = ndata.iloc[:,1:3].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled_columns = ['power_30s_avr', 'speed_wind_30s_avr']

# 将序列数据转化为监督学习数据
reframed = series_to_supervised(scaled, scaled_columns, 1, 1)
# 只将power作为y
reframed.drop(reframed.columns[[3]], axis=1, inplace=True)

values = reframed.values


n_trian_hours = 24*120*15
train = values[:n_trian_hours,:]
test = values[n_trian_hours:,:]
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
train_X = train_x.reshape((train_x.shape[0],1, train_x.shape[1]))
test_X = test_x.reshape(( test_x.shape[0],1, test_x.shape[1]))

model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y))

'''
    对数据绘图
'''
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make the prediction,为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
yHat = model.predict(test_X)

'''
    这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
'''
inv_yHat = concatenate((yHat, test_x[:, 1:]), axis=1)   # 数组拼接
inv_yHat = scaler.inverse_transform(inv_yHat)
inv_yHat = inv_yHat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)    # 将标准化的数据转化为原来的范围
inv_y = inv_y[:, 0]

rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
print('Test RMSE: %.3f' % rmse)
