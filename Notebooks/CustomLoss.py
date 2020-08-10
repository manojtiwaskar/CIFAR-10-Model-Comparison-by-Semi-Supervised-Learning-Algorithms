import tensorflow as tf
tf.enable_eager_execution()
import keras.backend as K


def dice_loss(y_true, y_pred):
    Y = K.flatten(y_true)
    A = K.flatten(y_pred)
    Y = K.print_tensor(Y, message="Y is: ")
    A = K.print_tensor(A, message="A is: ")
    print(A.numpy())
    # I = K.argmax(Y)
    # I = K.flatten(I)
    # print(I)
    # Get this shit working
    # I = K.print_tensor(I, message="I is: ")

    return (2.) / (K.sum(Y) + K.sum(A))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=dice_loss,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=2,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
