from __future__ import print_function

from pathlib import Path
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from keras import losses, optimizers,layers
from keras.layers import Conv2D, Flatten, BatchNormalization  # we have 2D images 
from keras.layers import Dense, ReLU 
from keras.models import Sequential 
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from  sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sbn
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from warnings import filterwarnings
from heapq import nlargest
from operator import itemgetter
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


                          #################################
                          #    BASIC CNN ARCHITECTURE     #
                          #################################
                          
# Set the seed value of the random number generator
#np.random.seed(2)
num_classes = 10
dataDict = sci.loadmat(r"NumberRecognitionBigger.mat")
print(dataDict.keys())

image_data = dataDict['X']
labels = dataDict['y']

#transpose the data
xTransposedData = np.transpose(image_data,[2,0,1])
xReshapedData = xTransposedData.reshape(30000,28,28,1)
yTransposedData = labels.transpose()
yReshapedData = yTransposedData.reshape(30000,1)

errorRate_knn_1 = []
accuracy_knn_1 = []
kFoldVal = KFold(n_splits=5, random_state=42, shuffle= True)
for trainIndex, testIndex in kFoldVal.split(xReshapedData):
    x_train, x_test, y_train, y_test = xReshapedData[trainIndex], xReshapedData[testIndex], yReshapedData[trainIndex], yReshapedData[testIndex]
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #model is build via sequential
    model = Sequential()
    model.add(BatchNormalization())
    # model.add(Conv2D(5, kernel_size=3, input_shape=(1,28,28,3), 
    #               activation="linear",  # only to match MATLAB defaults 
    #               data_format="channels_last"))
    
    model.add(Conv2D(5, kernel_size=3, input_shape=(28,28,1), 
                  activation="linear",  # only to match MATLAB defaults 
                  data_format="channels_last"))
  
    #model.build(input_shape=(28,28,1))
    model.add(ReLU())
    model.add(Flatten())
    model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(layers.Dropout(0.2)) #regularization term
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))


    model.add(Dense(units=10, activation="softmax") # 10 units, 10 digits 
    )  # multiclass classification output, use softmax

    #model.summary()   # Show a summary of the network architecture
    model.compile(optimizer=optimizers.SGD(momentum=0.9, lr=0.001), loss=losses.mean_squared_error, metrics=["accuracy"], )


    history = model.fit(x_train, y_train, epochs=15, verbose=1)
    y_pred = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('accuracy KNN', accuracy_knn_1)
                          


                          #################################
                          #    FLEXIBLE ARCHITECTURE      #
                          #################################


# Set the seed value of the random number generator
#np.random.seed(2)
num_classes = 10
dataDict = sci.loadmat(r"NumberRecognitionBigger.mat")
print(dataDict.keys())

image_data = dataDict['X']
labels = dataDict['y']

#transpose the data
xTransposedData = np.transpose(image_data,[2,0,1])
xReshapedData = xTransposedData.reshape(30000,28,28,1)
yTransposedData = labels.transpose()
yReshapedData = yTransposedData.reshape(30000,1)


kFoldVal = KFold(n_splits=5, random_state=42, shuffle= True)
for trainIndex, testIndex in kFoldVal.split(xReshapedData):
    x_train, x_test, y_train, y_test = xReshapedData[trainIndex], xReshapedData[testIndex], yReshapedData[trainIndex], yReshapedData[testIndex]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    #code reference : https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    #model is build via sequential
    model = Sequential()

    #add convolution layer
    model.add(Conv2D(32, kernel_size=3, input_shape=(28,28,1), activation="relu"))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=(28,28,1), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.4))

    #add convolution layer
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.4))
        
    # Finish with 10 softmax output nodes
    model.add(layers.Dense(num_classes, activation='softmax'))# multiclass classification output, use softmax
    model.summary()
    
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=15, verbose=1)
score = model.predict(x_test)
__file__ = str(Path(".").absolute() / "python_predict.py")
with open("python_predict.py", "r") as file:
    script = file.read()
    exec(script)