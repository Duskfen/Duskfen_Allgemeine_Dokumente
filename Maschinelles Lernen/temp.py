# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense #a wenn pylint sogt des geht ned... es geht
# from tensorflow.keras.optimizers import Adam
# import numpy as np

# model = Sequential([
#             Dense(16, input_shape=(1,), activation='relu'),
#             Dense(32, activation='relu'),
#             Dense(2, activation='sigmoid')
#         ])

# model.compile(
#         Adam(lr=0.0001),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#         )

# train_examples = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# labels=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) 

# model.fit(
#         train_examples,
#         labels,
#         batch_size=2,
#         epochs=20,
#         shuffle=True,
#         verbose=2
#         )

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(16, input_shape=(2,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])

model.compile(
    Adam(lr=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_samples = [
    [150, 67], 
    [130, 60], 
    [200, 65], 
    [125, 52], 
    [230, 72], 
    [181, 70]
]
# 0: male
# 1: female
train_labels = [1, 1, 0, 1, 0, 0]

model.fit(
    x=train_samples,
    y=train_labels,
    batch_size=3,
    epochs=10,
    shuffle=True,
    verbose=2
)