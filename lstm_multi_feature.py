from data import loaddata
from config import RAW_DATA, PROCESS_LEVEL1
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

ndata = data[['power_30s_avr', 'speed_wind_30s_avr', 'temp_de', 'speed_generator', 'temp_nde',
              'speed_rotor', 'speed_high_shaft', 'temp_ambient', 'temp_main_bearing']]



def model_train_and_fit(samples_num=20000, n_in=10, epochs=25, batch_size=32):
    values = ndata.iloc[:20000, :].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y = scaled[:, 0:1]
    scaled_exc_y = scaled[:, 1:]
    scaled_columns = ['speed_wind_30s_avr', 'temp_de', 'speed_generator', 'temp_nde', 'speed_rotor', 'speed_high_shaft',
                      'temp_ambient', 'temp_main_bearing']

    # 将序列数据转化为监督学习数据
    """
           Frame a time series as a supervised learning dataset.
           Arguments:
               data: Sequence of observations as a list or NumPy array.
               n_in: Number of lag observations as input (X).
               n_out: Number of observations as output (y).
               dropnan: Boolean whether or not to drop rows with NaN values.
           Returns:
               Pandas DataFrame of series framed for supervised learning.
       """
    n_in = 10
    # 调整n_in 即可

    # 将序列数据转化为监督学习数据
    reframed = series_to_supervised(scaled_exc_y, scaled_columns, n_in, 0)

    # 对齐powert 与  t-5的数据
    scaled_y = scaled[:-n_in, 0:1]
    reframed['power(t)'] = scaled_y

    values = reframed.values

    # 划分训练集和测试集
    train_size = round(len(values) * 0.67)
    train = values[:train_size, :]
    test = values[train_size:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
    train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    model = Sequential()
    # model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=25, batch_size=32)
    # , validation_data=(test_X, test_y)

    '''
           对数据绘图
       '''
    plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make the prediction,为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
    yHat = model.predict(test_X)

    '''
           这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
       '''
    inv_yHat = concatenate((yHat, test_x[:, :8]), axis=1)  # 数组拼接
    inv_yHat = scaler.inverse_transform(inv_yHat)
    inv_yHat = inv_yHat[:, 0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, :8]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)  # 将标准化的数据转化为原来的范围
    inv_y = inv_y[:, 0]

    rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
    print('Test RMSE: %.3f' % rmse)

    ahead_second = n_in * 30

    plt.figure(12)
    plt.suptitle("%s s ahead,Test RMSE:%s" % (ahead_second, rmse))
    plt.subplot(221), plt.plot(inv_yHat, label='predict')
    plt.legend()
    plt.subplot(223), plt.plot(inv_y, label='raw')
    plt.legend()
    plt.subplot(122), plt.plot(inv_y, label='raw'), plt.plot(inv_yHat, label='predict')
    plt.legend()
    plt.show()

model_train_and_fit(20000, 20, 25, 32)