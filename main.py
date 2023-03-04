import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tqdm import tqdm

def preprocess(data, steps):
    features = []
    labels = []
    for col in tqdm(data.columns):
        vals = data.loc[:,col].dropna()
        vals = vals.loc[vals!=0.0]
        vals = (vals - vals.min()) / (vals.max() - vals.min())
        for i in range(len(vals) - steps - 1):
            feature = vals.iloc[i:i+steps].values
            label = vals.iloc[i+steps+1]
            if label != 0.0:
                features.append(feature)
                labels.append(label)
    features, labels = np.array(features), np.array(labels)
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    return features, labels

def data_split(features_set, labels):
    if len(features_set) != len(labels):
        raise ValueError("Feature set must be the same length as labels")
    l = len(labels)
    training_split = int(l * 0.8)
    f_train  = features_set[:training_split]
    f_test = features_set[training_split:]
    l_train = labels[:training_split]
    l_test = labels[training_split:]
    return f_train, l_train, f_test, l_test

print("Preprocess data")
data = pd.read_csv("sp500_data.csv").iloc[:,1:]
features_set, labels = preprocess(data, 60)
f_train, l_train, f_test, l_test = data_split(features_set, labels)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Create model")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(f_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Train model")
model.fit(f_train, l_train, epochs=100, batch_size=64, validation_split=0.1)

print("Evaluate on test data")
results = model.evaluate(f_test, l_test, batch_size=128)
print("test loss, test acc:", results)



