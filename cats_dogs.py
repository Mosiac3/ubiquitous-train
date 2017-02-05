import os,random
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input,Dropout,Dense,Flatten,Convolution2D,MaxPooling2D,Activation
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.optimizers import RMSprop
from keras.utils import np_utils

train_dir = 'train/'
test_dir = 'test/'

rows = 64
cols = 64
channels = 3

train_images = [train_dir + i for i in os.listdir(train_dir)]
train_dogs = [train_dir + i for i in os.listdir(train_dir) if 'dog' in i]
train_cats = [train_dir + i for i in os.listdir(train_dir) if 'cat' in i]

test_images = [test_dir + i for i in os.listdir(test_dir)]

# slice datasets for memory efficency
train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)
test_images = test_images[:25]

# read images
def read_images(file_path):
  img = cv2.imread(file_path,cv2.IMREAD_COLOR)
  return cv2.resize(img,(rows,cols),interpolation=cv2.INTER_CUBIC)

# prepare train/test data
def prep_data(images):
  count = len(images)
  data = np.ndarray((count,channels,rows,cols),dtype=np.uint8)

  for i, image_file in enumerate(images):
	image = read_images(image_file)
	data[i] = image.T
  
  return data

train = prep_data(train_images)
test = prep_data(test_images)

print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))

# create labels
labels = []
for i in train_images:
  if 'dog' in i:
	labels.append(1)
  else:
	labels.append(0)

# find out label distribution
#sns.countplot(labels)
#sns.plt.title('Cats and Dogs')

# display first 5 cats and dog images 
def show_cats_and_dogs(idx):
  cat = read_images(train_cats[idx])
  dog = read_images(train_dogs[idx])
  pair = np.concatenate((cat, dog),axis=1)
  plt.figure(figsize=(10,5))
  plt.imshow(pair)

#for idx in range(0,5):
 # show_cats_and_dogs(idx)

# cnn model
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def catdog():
  model = Sequential()

  model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,rows,cols),activation='relu'))
  model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))

  model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
  model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))

  model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
  model.add(Convolution2D(128,3,3,border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))

  model.add(Convolution2D(256,3,3,border_mode='same',activation='relu'))
  model.add(Convolution2D(256,3,3,border_mode='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))

  model.add(Flatten())
  model.add(Dense(256,activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(256,activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.summary()

  model.compile(loss=objective,optimizer=optimizer,metrics=['accuracy'])
  return model

model = catdog()

# train and predict
nb_epoch = 10
batch_size = 16

class LossHistory(Callback):
  def on_train_begin(self,logs={}):
	self.losses=[]
	self.val_losses=[]
  def on_epoch_end(self,batch,logs={}):
	self.losses.append(logs.get('loss'))
	self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss',patience=3,verbose=1,mode='auto')

def run_catdog():
  history = LossHistory()
  model.fit(train,labels,batch_size = batch_size, nb_epoch=nb_epoch,validation_split=0.25,verbose=0,shuffle=True,callbacks=[history,early_stopping])
  predictions = model.predict(test,verbose=0)
  return predictions, history

predictions,history = run_catdog()

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss,'blue',label='Training Loss')
plt.plot(val_loss,'green',label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()

# how good is the prediction?
for i in range(0,10):
  if predictions[i,0] >=0.5:
	print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
  else:
	print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

  plt.imshow(test[i].T)
  plt.show()

