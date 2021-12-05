#THIS IS A GROUP PROJECT FOR CS 334. CONTRIBUTORS INCLUDE: ENDER SHMIDT, ABDULLAH HAMID AND TULIO CANO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
df  = pd.read_csv('CutDataSet.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
train_set = df_train.values
train_set = np.reshape(train_set, (len(train_set), 1))
train_set = train_set.astype("float32")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)
X_train = train_set[0:len(train_set)-1]
y_train = train_set[1:len(train_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))


# In[47]:


model = Sequential()
model.add(LSTM(units = 10, activation = 'relu', input_shape = (None, 1)))
model.add(Dense(units = 1))
model.compile(loss = 'mean_squared_error',optimizer = 'adam',)
model.fit(X_train, y_train, batch_size = 32, epochs = 100)


# In[48]:


test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = model.predict(inputs)
predicted_BTC_price = scaler.inverse_transform(predicted_BTC_price)


# In[49]:


plt.figure(figsize=(20,8), dpi=78, edgecolor='b')
ax = plt.gca()
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price by our Model')
plt.plot(test_set, color = 'red', label = 'Actual BTC Price')
plt.title('Bitcoin Price Prediction using LSTM', fontsize=34)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = '60')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(12)
plt.xlabel('Time', fontsize=24)
plt.ylabel('BTC Price', fontsize=24)
plt.legend(loc=2, prop={'size': 14})
plt.show()


# In[ ]:





# In[ ]:




