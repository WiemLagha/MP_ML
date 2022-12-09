import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

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
import asyncio

def trainig(filename):
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
    model_fit= model.fit(train_dataset, steps_per_epoch=3, epochs=1, validation_data=test_dataset)
    img= load_img(filename,target_size = (200,200))
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
    


def upload_file():
    f_types=[('Jpg files','*.jpg'),('PNG files', '*.png')]
    filename=tk.filedialog.askopenfilename(filetypes=f_types)
    img=ImageTk.PhotoImage(file=filename)
    e1=tk.Label(my_w)
    e1.grid(row=3, column=1)
    e1.image=img
    e1['image']=img
    asyncio.run_coroutine_threadsafe(trainig(filename))
    
    
    
    

my_w= tk.Tk()
my_w.geometry('410x300')
my_w.title ("import image")
my_w.font1=('times',18,'bold')

l1=tk.Label(my_w,text='Upload files', width=50)
l1.grid(row=1,columnspan=4)

b1=tk.Button(my_w,text='Upload image', width=20, command=lambda:upload_file())
b1.grid(row=2, column=1,columnspan=4)
my_w.mainloop()



