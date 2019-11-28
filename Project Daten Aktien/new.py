#total crap.. Gabriel, BITTE VERSUUUUCH NACHZUDENKEN XD.....

import pandas
import numpy as np
import xlrd
from numpy import array

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

training_data_x = pandas.read_excel("D:/Documents/Allgemeine_Dokumente/Project Daten Aktien/test.xlsx")
# raw = training_data_x.as_matrix()
raw = np.array(training_data_x)
# print(raw)

train_samples = []
train_labels = []


print(float(raw[0]))

x=-1
for i in raw:
    x += 1
    if x%2 == 0:
        train_samples.append(float(raw[x]))
    else:
        train_labels.append(float(raw[x]))

train_samples = array(train_samples)
train_labels = array(train_labels)

print(train_labels[0])
print(train_samples[0])


model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='softmax')
])

model.compile(
    Adam(lr=0.0001),
    loss='kullback_leibler_divergence',
    metrics=['accuracy']
    
)

model.fit(
    x=train_samples,
    y=train_labels,
    batch_size=200,
    epochs=30,
    shuffle=True,
    verbose=2
)

scaled_test_samples= raw[0]

predictions = model.predict(
    scaled_test_samples, 
    batch_size=100, 
    verbose=0
)

print(predictions)

