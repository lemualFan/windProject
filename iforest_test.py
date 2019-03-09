print(__doc__)

from data import loaddata
from config import RAW_DATA, PROCESS_LEVEL1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()


data = loaddata.load_data(RAW_DATA) 
data.to_csv(PROCESS_LEVEL1)
ndata = data[['power_30s_avr', 'speed_wind_30s_avr', 'temp_de', 'speed_generator', 'temp_nde','speed_rotor', 'speed_high_shaft', 'temp_ambient', 'temp_main_bearing']]
rng = np.random.RandomState(42)
clf = IsolationForest(behaviour='new', max_samples=100, random_state=rng, contamination='auto')
X_train = ndata.iloc[:2000,:]
X_test = ndata.iloc[2000:3000,:]
clf.fit(X_train)
y_pred_train = clf.predict(X_train) 
y_pred_test = clf.predict(X_test)
y_pred_test_pd = pd.DataFrame(y_pred_test)                                                                                                                             

y_pred_train_pd = pd.DataFrame(y_pred_train)
#将结果与
res_train = pd.concat([X_train,y_pred_train_pd],axis=1)