import numpy as np
import pandas as pd
#import seaborn as sns
import scipy.stats as sp
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Lambda
import tensorflow as tf

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential

from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_path = 'vmtable.csv'
headers=['vmid','subscriptionid','deploymentid','vmcreated', 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu', 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']
trace_dataframe = pd.read_csv(data_path, header=None, index_col=False,names=headers,delimiter=',')
df = pd.DataFrame(trace_dataframe, columns = ['maxcpu', 'avgcpu'])
df['mincpu'] = 2*df['avgcpu'] - df['maxcpu']
df

dataset = df.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


train_size = int(len(dataset)* 0.8)
test_size = len(dataset) - train_size 
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_training_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a= dataset[i:(i+look_back), :3]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


look_back = 40
trainX, trainY = create_training_dataset(train, look_back=look_back)
testX, testY = create_training_dataset(test, look_back=look_back)
# Vérifiez les données pour des valeurs invalides
assert not np.any(np.isnan(trainX)), "trainX contient des NaN"
assert not np.any(np.isnan(trainY)), "trainY contient des NaN"
assert not np.any(np.isinf(trainX)), "trainX contient des valeurs infinies"
assert not np.any(np.isinf(trainY)), "trainY contient des valeurs infinies"


adamOpt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    decay=0.0,
    amsgrad=False
)


model_Autoencoder = Sequential()
model_Autoencoder.add(LSTM(units=64, input_shape=(trainX.shape[1], trainX.shape[2]))) 
model_Autoencoder.add(Dropout (rate=0.2))
model_Autoencoder.add(tf.keras.layers.RepeatVector (n=trainX.shape[1]))
model_Autoencoder.add(LSTM(units=64, return_sequences=True))
model_Autoencoder.add(Dropout (rate=0.2))
model_Autoencoder.add(tf.keras.layers.TimeDistributed (tf.keras.layers.Dense (units=trainX.shape[2])))
model_Autoencoder.compile(loss='mae', optimizer='adam')
history_Autoencoder = model_Autoencoder.fit(trainX, trainX, validation_split=0.25, epochs=50


model_Autoencoder.summary()


from sklearn.metrics import mean_squared_error, mean_absolute_error
# Faites des prédictions
trainPredict_A = model_Autoencoder.predict(trainX)
testPredict_A = model_Autoencoder.predict(testX)
# Inverser les prédictions


print('Shape of trainPredict:', trainPredict_A.shape)
print('Shape of trainY:', trainY.shape)
print('Shape of testPredict:', testPredict_A.shape)
print('Shape of testY:', testY.shape)


# Ajuster et transformer les prédictions
trainPredict_A_flat = trainPredict_A.reshape(-1, trainPredict_A.shape[-1])
testPredict_A_flat = testPredict_A.reshape(-1, testPredict_A.shape[-1])
trainPredict_A_inverse = scaler.inverse_transform(trainPredict_A_flat).reshape(trainPredict_A.shape)
testPredict_A_inverse = scaler.inverse_transform(testPredict_A_flat).reshape(testPredict_A.shape)

trainY_reshaped = trainY.reshape(-1, trainY.shape[-1])  # Assurez-vous que trainY a la dimension correcte pour inverse_transform
trainY_inversed = scaler.inverse_transform(trainY_reshaped).flatten()
testY_reshaped = testY.reshape(-1, testY.shape[-1])  # Assurez-vous que trainY a la dimension correcte pour inverse_transform
testY_inversed = scaler.inverse_transform(testY_reshaped).flatten()

# Calculer l'erreur quadratique moyenne racine (RMSE)
trainScore_A = np.sqrt(mean_squared_error(trainY_inversed, trainPredict_A_inverse.flatten()))
print('Train Score: %.2f RMSE' % trainScore_A)

testScore_A = np.sqrt(mean_squared_error(testY_inversed, testPredict_A_inverse.reshape(-1, 3)))
print('Test Score: %.2f RMSE' % testScore_A)


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict_A
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict_A)+(look_back*2)+1:len(dataset)-1, :] = testPredict_A
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
