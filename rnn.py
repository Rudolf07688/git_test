# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

# FRL Functions
# -------------------------------------- #
def get_delta(data):
    xDelta = np.zeros((0, 60))
    for i in range(0,data.shape[0]):#tot 7140
        row = []
        row.append(0)
        for j in range(1, data.shape[1]): #tot 60
            row.append(data[i, j] - data[i, j-1])
        row = np.array(row).reshape((1,60))
        xDelta = np.vstack([xDelta, row])   
        
    return xDelta
            
def get_random_offset_seq(seq_len, max_offset=0.5):
    line = []
    while (len(line) < seq_len):
        length = np.random.randint(400,1000)
        coin = np.random.uniform(0,1)
        if (coin > 0.5):
            offset = 0
        else:
            offset = np.random.uniform(low=0, high=max_offset)
        for i in range(0, length):
            if (len(line) >= seq_len):
                break
            else:
                line.append(offset)
    return line

def get_random_skew_offset_seq(seq_len, max_offset=1):
    line = np.array([])
    while (line.shape[0] < seq_len):
        length = np.random.randint(600,1000)
        coin = np.random.uniform(0,1)
        if (coin > 0.8):
            offset = 0
        else:
            offset = np.random.uniform(low=-0.5, high=max_offset)
        
        t = np.linspace(0, offset, num=length)
        line = np.concatenate([line, t])
        
    while (line.shape[0] > seq_len):
        line = np.delete(line, -1)
            
    return line

def gen_random_frequencies(seq_len=4200):
    min_len = 200
    max_len = 500
    max_freq = 3
    
    line = np.array([])
    while (line.shape[0] < seq_len):
        length = np.random.randint(min_len, max_len)
        t = np.linspace(1, max_freq, length)
        line = np.append(line, t)
        
    while (line.shape[0] > seq_len):
        line = np.delete(line, -1)
        
    return line

def split_sets(x_scaled, Y, train_set_size, test_set_size, x_scaled2 = np.array([0,0])):
    xTrain = x_scaled[:train_set_size, :]
    yTrain = Y[:train_set_size, :]
    #xTrain = np.reshape(xTrain, (xTrain.shape[0],xTrain.shape[1],1))
    
    xTest = x_scaled[train_set_size:train_set_size+test_set_size, :]
    #xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
    yTest = Y[train_set_size:train_set_size+test_set_size, :]
    
    if (x_scaled2.any() != 0):
        xTrain2 = x_scaled2[:train_set_size, :]
        xTrain2 = np.reshape(xTrain2, (xTrain2.shape[0],xTrain2.shape[1],1))
        xTest2 = x_scaled2[train_set_size:train_set_size+test_set_size, :]
        xTest2 = np.reshape(xTest2, (xTest2.shape[0], xTest2.shape[1], 1))
        
        return xTrain, xTest, yTrain, yTest, xTrain2, xTest2
    
    return xTrain, xTest, yTrain, yTest
# -------------------------------------- #

# Ander dataset (LSTM_create_dataset_2.py)
x_set = np.array(pd.read_csv('LSTM_xSet_3_T.csv', header=None))
y_set = np.array(pd.read_csv('LSTM_ySet_3.csv', header=None))

x_set = x_set[5000:20000, :]
y_set = y_set[5000:20000, :]

train_size = 7200
test_size = 1200
xTrain, xTest, yTrain, yTest = split_sets(x_set, y_set, train_set_size=train_size, test_set_size=test_size)

# Introduce gaussian noise
train_noise = np.random.normal(0, 0.1*0.1, yTrain.shape)
test_noise = np.random.normal(0, 0.1*0.1, yTest.shape)
yTrain = yTrain + train_noise
yTest = yTest + test_noise

xTrain = []
yy_yTrain = []
xTest = []
yy_yTest = []
for i in range(60, train_size):
    xTrain.append(yTrain[i-60:i, 0])
    yy_yTrain.append(yTrain[i, 0])
    if (i < test_size):
        xTest.append(yTest[i-60:i, 0])
        yy_yTest.append(yTest[i, 0])
xTrain, yy_yTrain = np.array(xTrain), np.array(yy_yTrain)
xTest, yy_yTest = np.array(xTest), np.array(yy_yTest)

fr_sc_x = MinMaxScaler(feature_range = (-1, 1))
fr_sc_y = MinMaxScaler(feature_range = (0, 1))



# Reshaping

# -------------------------------------------------------#




Import = 1

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# FRL schenanigans
# ---------------------------------------------------------------------------- #
# Creating dataset
train_size = 7200
test_size = 1200
noise_max = 0.1
#frl_set = 5*np.cos(t*np.pi*2) + 7*np.sin((t+1.13)*np.pi*5)+noise+offset_line + 11
if (Import == 0):
    offset_line = get_random_offset_seq(train_size+test_size, 8)
    skew_offset = get_random_skew_offset_seq(train_size+test_size, 4)
    t = np.linspace(0,160,train_size+test_size)
    noise = np.random.uniform(low=-noise_max, high=noise_max, size=(train_size+test_size))
    rand_freqs = gen_random_frequencies(seq_len=t.shape[0])
    
    frl_set = 0.2*np.cos(t*np.pi/rand_freqs) + 0.4*np.sin((t+1.13)*np.pi*2)+noise+offset_line+skew_offset + 12
elif (Import == 1):
    frl_set = np.array(pd.read_csv('frl_synth_data.csv', header=None))
    frl_set = np.reshape(frl_set, (frl_set.shape[0], 1))


frl_set = np.round(frl_set, decimals=1)
frl_test_data = frl_set[train_size:].reshape((test_size,1))
frl_set = frl_set[:train_size].reshape((train_size, 1))

#frl_set_scaled = frl_sc.fit_transform(frl_set)
#frl_test_scaled = frl_sc.fit_transform(frl_test_data)

# Creating a data structure with 60 timesteps and 1 output #
frl_x_train = []
frl_y_train = []
frl_x_test = []
frl_y_test = []
for i in range(60, train_size):
    frl_x_train.append(frl_set[i-60:i, 0])
    frl_y_train.append(frl_set[i, 0])
    if (i < test_size):
        frl_x_test.append(frl_test_data[i-60:i, 0])
        frl_y_test.append(frl_test_data[i, 0])
frl_x_train, frl_y_train = np.array(frl_x_train), np.array(frl_y_train)
frl_x_test, frl_y_test = np.array(frl_x_test), np.array(frl_y_test)

# Getting the deltas #
frl_x_train_delta = get_delta(frl_x_train)
frl_x_test_delta = get_delta(frl_x_test)
# Getting the delta_deltas
frl_x_train_delta_delta = get_delta(frl_x_train_delta)
frl_x_test_delta_delta = get_delta(frl_x_test_delta)


# Stacking the deltas horizontally with the training data #
#frl_x_train = np.hstack([frl_x_train, frl_x_train_delta])
#frl_x_test = np.hstack([frl_x_test, frl_x_test_delta])

# Delta deltas
#frl_x_train = np.hstack([frl_x_train, frl_x_train_delta_delta])   
#frl_x_test = np.hstack([frl_x_test, frl_x_test_delta_delta])
# ------------------------------------------------------- #


# Feature Scaling #
frl_sc = MinMaxScaler(feature_range = (-1, 1))

frl_x_train = frl_sc.fit_transform(frl_x_train)
frl_x_test = frl_sc.fit_transform(frl_x_test)

# Regte data
xTrain = frl_sc.fit_transform(xTrain)
xTest = frl_sc.fit_transform(xTest)
# ---------------------------------------------- #


# Delta feature scaling
frl_x_train_delta = frl_sc.fit_transform(frl_x_train_delta)
frl_x_test_delta = frl_sc.fit_transform(frl_x_test_delta)

frl_x_train_delta_delta = frl_sc.fit_transform(frl_x_train_delta_delta)
frl_x_test_delta_delta = frl_sc.fit_transform(frl_x_test_delta_delta)
# ----------------------------------------------- #



# Reshaping
frl_x_train = np.reshape(frl_x_train, (frl_x_train.shape[0], frl_x_train.shape[1], 1))
frl_x_test = np.reshape(frl_x_test, (frl_x_test.shape[0], frl_x_test.shape[1], 1))

# Regte data
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
# ---------------------------------------------------------------------------- #

# Stacking the deltas vertically with the training data #
# Reshape first
frl_x_train_delta = np.reshape(frl_x_train_delta_delta, (frl_x_train_delta_delta.shape[0], frl_x_train_delta_delta.shape[1], 1))
frl_x_test_delta = np.reshape(frl_x_test_delta, (frl_x_test_delta.shape[0], frl_x_test_delta.shape[1], 1))
frl_x_train_delta_delta = np.reshape(frl_x_train_delta_delta, (frl_x_train_delta_delta.shape[0], frl_x_train_delta_delta.shape[1], 1))
frl_x_test_delta_delta = np.reshape(frl_x_test_delta_delta, (frl_x_test_delta_delta.shape[0], frl_x_test_delta_delta.shape[1], 1))

# Adding deltas
frl_x_train = np.concatenate((frl_x_train, frl_x_train_delta), axis=-1)
frl_x_test = np.concatenate((frl_x_test, frl_x_test_delta), axis=-1)
# Adding delta deltas
#frl_x_train = np.concatenate((frl_x_train, frl_x_train_delta_delta), axis=-1)
#frl_x_test = np.concatenate((frl_x_test, frl_x_test_delta_delta), axis=-1)
# ----------------------------------------------------- #


#sys.exit("FRL")

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (frl_x_train.shape[1], frl_x_train.shape[-1])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1, activation='relu'))

# Compiling the RNN
adam = optimizers.Adam(lr=0.001, clipnorm=1.)
sgd = optimizers.SGD(lr=0.01, momentum=0.75, nesterov=True, clipnorm=1.)
regressor.compile(optimizer = sgd, loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(frl_x_train, frl_y_train, epochs = 1, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
#real_stock_price = dataset_test.iloc[:, 1:2].values
#
## Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#inputs = inputs.reshape(-1,1)
#inputs = sc.transform(inputs)
#X_test = []
#for i in range(60, 80):
#    X_test.append(inputs[i-60:i, 0])
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#predicted_stock_price = regressor.predict(X_test)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predictions = regressor.predict(frl_x_test)

#predicted_stock_price = frl_sc.inverse_transform(predicted_stock_price)
# Visualising the results
plt.plot(yTest, color='black', label='Real test set examples')
plt.plot(frl_y_test, color = 'red', label = 'ySet Values')
plt.plot(predictions, color = 'blue', label = 'Predicted Values')
plt.title('LSTM test')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

frl_y_test = frl_y_test.reshape((frl_y_test.shape[0], 1))
error = np.average(np.abs(predictions - frl_y_test)/np.abs(frl_y_test))*100
print("Average error: {:.2f} %".format(error))
