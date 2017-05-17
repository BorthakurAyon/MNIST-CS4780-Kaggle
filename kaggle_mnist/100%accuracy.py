from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy

x_train = numpy.loadtxt("trainX.csv", delimiter=",")
y_train = numpy.loadtxt("trainY.csv", delimiter=",")
x_test = numpy.loadtxt("testX.csv", delimiter=",")

# Shuffling
idx = numpy.random.choice(4000, 4000, replace=False)

x_train = x_train[idx, :]
y_train = y_train[idx, :]

x_val = x_train[3600:4001, :]
y_val = y_train[3600:4001, :]

x_train = x_train[0:3600, :]
y_train = y_train[0:3600, :]


# scaling
x_train = x_train/255
x_val = x_val/255
x_test = x_test/255

# Reshaping
# import pdb;pdb.set_trace()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
num_classes = y_val.shape[1]

datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2)

datagen.fit(x_train)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150,
          # batch_size=200, verbose=2)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=200), validation_data=(x_val, y_val),
                    steps_per_epoch=200, epochs=400, verbose=2)


# Predict
y_pred = model.predict(x_test)
rounded = [numpy.round(x) for x in y_pred]
# print(numpy.shape(rounded))

y_test = numpy.zeros((800, 1))

for i in range(800):
    # import pdb;pdb.set_trace()
    val, = numpy.nonzero(rounded[i][:])
    if bool(val):
        # print(val)
        y_test[i, 0] = val
    else:
        y_test[i, 0] = 0
y_test = numpy.int_(y_test)
# import pdb;pdb.set_trace()
# header = ["id", "digit"]
df = pd.DataFrame(y_test)
df.to_csv("y_test.csv")
# print(y_test)
