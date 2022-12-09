from tensorflow.python import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

from keras.utils import load_img
from keras.utils import img_to_array

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


train = ImageDataGenerator(rescale=1/255)
test= ImageDataGenerator(rescale= 1/255)
train_dataset= train.flow_from_directory("./dataset/training", target_size=(200,200), batch_size=3, class_mode= "binary")
test_dataset=train.flow_from_directory("./dataset/test", target_size=(200,200), batch_size=3, class_mode= "binary")

model= tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(200,200,3)),
tf.keras.layers.MaxPool2D(2,2),
##
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
##
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
##
tf.keras.layers.Flatten(),
##
tf.keras.layers.Dense(512, activation='relu'),
##
tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer= RMSprop(lr=0.01), metrics=['accuracy'])
model_fit= model.fit(train_dataset, steps_per_epoch=3, epochs=10, validation_data=test_dataset)

dir_path= './dataset/prediction'
for i in os.listdir(dir_path):
    img= load_img(dir_path+'\\'+i,target_size = (200,200))
    X= img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images= np.vstack([X])

    val= model.predict(images)
    if (val==0):
        print ('normal \n')
    elif (val == 1 ):
        print('begnin \n')
    else:
        print('malin \n')


