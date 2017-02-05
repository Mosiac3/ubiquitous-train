import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.utils import np_utils

# import data
train = pd.read_csv('train.csv').values
test = pd.read_csv('test.csv').values

nb_epoch = 1
batch_size = 128
img_rows,img_cols = 28,28

# define trainX
trainX = train[:,1:].reshape(train.shape[0],img_rows,img_cols,1)
trainX = trainX.astype(float)
trainX /= 255.0

# define trainY
trainY = np_utils.to_categorical(train[:,0])
nb_classes = trainY.shape[1]

# Sequential model
cnn = Sequential()

cnn.add(Convolution2D(32,3,3,activation='relu',input_shape=(28,28,1),border_mode='same'))
cnn.add(Convolution2D(32,3,3,activation='relu',border_mode='same'))
cnn.add(MaxPooling2D(strides=(2,2)))

cnn.add(Convolution2D(64,3,3,activation='relu',border_mode='same'))
cnn.add(Convolution2D(64,3,3,activation='relu',border_mode='same'))
cnn.add(MaxPooling2D(strides=(2,2)))

cnn.add(Flatten())
cnn.add(Dropout(0.2))
cnn.add(Dense(128,activation='relu'))
cnn.add(Dense(10,activation='softmax'))

cnn.summary()

# compile
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# fit model
cnn.fit(trainX,trainY,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1)

# testX
testX = test.reshape(test.shape[0],28,28,1)
testX = testX.astype(float)
testX /= 255.0

# predict testY
yPred = cnn.predict_classes(testX)

# save to file
np.savetxt('mnist-vggnet.csv',np.c_[range(1,len(yPred)+1),yPred],delimiter=',',header='ImageId,Label',comments ='',fmt='%d')

