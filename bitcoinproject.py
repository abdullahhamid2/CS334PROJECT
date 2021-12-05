#THIS IS A GROUP PROJECT FOR CS 334. CONTRIBUTORS INCLUDE: ENDER SHMIDT, ABDULLAH HAMID AND TULIO CANO
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import SimpleRNN
import rnn


def preprocessing(df, timestep):
    xTrain=[]
    yTrain=[]
    df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
    group = df.groupby('date')
    #closevalues = group["Close"].mean()
    closevalues = group["Weighted_Price"].mean()
    length_close_values = len(closevalues)
    predictiondays = 50
    close_train=closevalues.iloc[:length_close_values-predictiondays]
    close_test=closevalues.iloc[len(close_train):]
    close_train = np.array(close_train)
    close_train = close_train.reshape(close_train.shape[0], 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    close_scaled = scaler.fit_transform(close_train)
    finalrange = close_scaled.shape[0]
    for i in range(timestep,finalrange):
        xTrain.append(close_scaled[i-timestep:i,0])
        yTrain.append(close_scaled[i,0])

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    return xTrain, yTrain, close_test, close_train, closevalues


def to_scale(close_train):
    close_train=np.array(close_train)
    close_train=close_train.reshape(close_train.shape[0],1)
    scaler=MinMaxScaler(feature_range=(0,1))
    close_scaled=scaler.fit_transform(close_train)
    return close_scaled, scaler

def lstm_model(xTrain, yTrain, closevalues, close_test, timestep,scaler):
    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(xTrain,yTrain,epochs=100,batch_size=32)

    inputs=closevalues[len(closevalues)-len(close_test)-timestep:]
    inputs=inputs.values.reshape(-1,1)
    inputs=scaler.transform(inputs)

    x_test=[]
    for i in range(timestep,inputs.shape[0]):
        x_test.append(inputs[i-timestep:i,0])
    x_test=np.array(x_test)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

    predicted_data=model.predict(x_test)
    predicted_data=scaler.inverse_transform(predicted_data)
    return predicted_data

# def rnn_model(xTrain, yTrain, closevalues, close_test,timestep,scaler):
    # yTrainOne = closevalues[0:len(yTrain)-1]
    # yTrainTwo = closevalues[1:len(yTrain)]
    # yTrainOne = np.reshape(yTrainOne, (yTrainOne.shape[0], yTrainOne.shape[1], 1))
    # regressor = Sequential()
    # function = "relu"
    # regressor.add(SimpleRNN(units=200, activation=function, return_sequences=True, input_shape=(1, 1)))
    # regressor.add(SimpleRNN(units=200))
    # regressor.add(Dense(units=1))
    # regressor.compile(optimizer='adam', loss='logcosh')
    # regressor.fit(yTrainOne, yTrainTwo, epochs=100, batch_size=200)
    # #yTestInput = np.reshape(close_test, (close_test.shape[0], 1))
    # yTestInput = scaler.transform(close_test)
    # yTestPred = regressor.predict(yTestInput)
    # yTestPred = scaler.inverse_transform(yTestPred)
    # return yTestPred

def grapher(predicted_data, close_test):
    data_test=np.array(close_test)
    data_test=data_test.reshape(len(data_test),1)

    plt.figure(figsize=(20,8), dpi=78, edgecolor='b')
    plt.plot(data_test,color="r",label="Actual Price of BITCOIN in US$")
    plt.plot(predicted_data,color="b",label="PREDICTED PRICE OF BITCOIN USING OUR LSTM MODEL")
    plt.legend()
    plt.xlabel("# of days = 50")
    plt.ylabel("Close Values in US$")
    plt.grid(True)
    plt.legend(loc=2, prop={'size': 14})
    plt.show()


def main():
    """
    Main file to run from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("timestep",
                        help="choose the amount of data as xTrain")
    args = parser.parse_args()
    timestep = int(args.timestep)
    df = pd.read_csv('CutDataSet.csv')
    xTrain, yTrain, close_test, close_train, closevalues = preprocessing(df, timestep)
    close_scaled,scaler = to_scale(close_train)
    predicted_data_lstm = lstm_model(xTrain, yTrain, closevalues, close_test, timestep, scaler)
    grapher(predicted_data_lstm, close_test)
    print("Running RNN Model now")
    rnn.main()



if __name__ == "__main__":
    main()