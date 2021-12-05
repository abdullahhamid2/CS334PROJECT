import argparse
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataSet", help = "Bitcoin dataset inserted in")
    # args = parser.parse_args()
    # dataSet = pd.read_csv(args.dataSet)
    dataSet = pd.read_csv('CutDataSet.csv')
    dataSet = dataSet.groupby("Timestamp", as_index = False).mean()
    ySet = dataSet["Weighted_Price"]   #closevalues
    ySet = np.array(ySet)
    trainSize = round(len(ySet) * 0.9)
    yTrain = ySet[:trainSize]
    yTest = ySet[trainSize:]
    scaler = MinMaxScaler()
    yTrain = yTrain.reshape(-1, 1)
    yTrain = scaler.fit_transform(yTrain)
    yTrainOne = yTrain[0:len(yTrain)-1]
    yTrainTwo = yTrain[1:len(yTrain)]
    yTrainOne = np.reshape(yTrainOne, (yTrainOne.shape[0], yTrainOne.shape[1], 1))
    regressor = Sequential()
    function = "relu"
    regressor.add(SimpleRNN(units = 200, activation=function, return_sequences = True, input_shape = (1, 1)))
    #Adds dropouts and extra RNN layers.
    #regressor.add(Dropout(0.2))
    #regressor.add(SimpleRNN(units = 32, activation=function, return_sequences = True))
    #regressor.add(Dropout(0.2))
    #regressor.add(SimpleRNN(units = 32, activation=function, return_sequences = True))
    #regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units = 200))
    #regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'logcosh')
    regressor.fit(yTrainOne, yTrainTwo, epochs = 100, batch_size = 200)
    yTestInput = np.reshape(yTest, (yTest.shape[0], 1))
    yTestInput = scaler.transform(yTestInput)
    yTestPred = regressor.predict(yTestInput)
    yTestPred = scaler.inverse_transform(yTestPred)
    plt.figure(figsize=(20,8), dpi=78, edgecolor='b')
    ax = plt.gca()  
    plt.plot(yTestPred, color = 'blue', label = 'Predicted BTC Price by our Model')
    plt.plot(yTest, color = 'red', label = 'Actual BTC Price')
    plt.title('Bitcoin Price Prediction using RNN', fontsize=34)
    x=range(len(ySet) - trainSize)
    labels = np.array(dataSet['Timestamp'])
    labels = labels[trainSize:]
    plt.xticks(np.arange(x[0], x[len(x) - 1], 50), labels[np.arange(x[0], x[len(x)-1], 50)], rotation = '60')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('BTC Price', fontsize=24)
    plt.legend(loc=2, prop={'size': 14})
    plt.show()
    print(explained_variance_score(yTest, yTestPred))
    print(r2_score(yTest, yTestPred))
    print(mean_squared_error(yTest, yTestPred))
    print(max_error(yTest, yTestPred))
    

if __name__ == "__main__":
    main()