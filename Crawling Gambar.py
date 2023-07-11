#!/usr/bin/env python
# coding: utf-8

# Import tensorflow
import tensorflow as tf
print(tf.__version__)

# Import library
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Install keras
get_ipython().system('pip install keras')

# Install tensorflow
get_ipython().system('pip install tensorflow')

# Install bing image downloder
pip install bing-image-downloader

# Download image
from bing_image_downloader import downloader
downloader.download('Ronaldo',limit=50,output_dir='image',adult_filter_off=True)

# Download image
downloader.download('Messi',limit=50,output_dir='image',adult_filter_off=True)

#Prepocessing training set
train_datagen=ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True)
training_set=train_datagen.flow_from_directory(
  'C:/Users/hendr/image',
  target_size=(64,64),
  batch_size=32,
  class_mode='binary')

#Prepocessing tes set
test_datagen=ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True)
test_set=test_datagen.flow_from_directory(
  'C:/Users/hendr/image',
  target_size=(64,64),
  batch_size=32,
  class_mode='binary')

# Make variable
cnn=tf.keras.models.Sequential()

# Layer 1 = Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

# Layer 2 = Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Layer3 = flattening
cnn.add(tf.keras.layers.Flatten())

# Layer 4 - full Conncetion
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Layer 5 = Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile model
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit model
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Make prediction
import numpy as np
from keras.preprocessing import image
# Load gambar yang akan diprediksi
test_image=tf.keras.utils.load_img('C:/Users/hendr/image/Ronaldo/image_1.jpg',target_size=(64,64))
test_image=tf.keras.utils.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
# Lakukan prediksi
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='Ronaldo'
else:
    prediction='Messi'

print (prediction)

'Terima kasih'

