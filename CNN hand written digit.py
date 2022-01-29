#!/usr/bin/env python
# coding: utf-8

# This is the hello world of deep learning, it is handwritten digit identificaiton

# In[1]:


import tensorflow as tf


# In[2]:


mnist = tf.keras.datasets.mnist# Automaticlly downloads mnist data set, default handwritten digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()# Train test split
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[3]:


print(x_train.shape)#input model top output bottom, this is the data to train model, image size =28x28
print(y_train.shape)#60000 entries of images  


# In[4]:


print(x_test.shape)#this is the data to test
print(y_test.shape)


# In[5]:


model = tf.keras.models.Sequential([# Builds and models
  tf.keras.layers.Flatten(input_shape=(28, 28)),# Keras flatten, flattens input shape of image
  tf.keras.layers.Dense(128, activation='relu'),# Implementation of perceptron, 
  tf.keras.layers.Dropout(0.2),# Randomly skips input values, drops values
  tf.keras.layers.Dense(10, activation='softmax')# 10 outputs betweem 0 and 1
])# Basic model for graph model in TF


# In[11]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)# Loss function
model.compile(optimizer = "adam", loss = loss_fn, metrics = ["accuracy"])


# In[12]:


model.fit(x_train, y_train, epochs=10)# Causes 10 outputs, training weight of graph


# In[8]:


model.evaluate(x_test, y_test)# Evaluates accuracy


# In[9]:


from PIL import Image
from IPython.display import display


# In[17]:


img = Image.open(r'C:\Enter your file path\Digit4.bmp')# This uploads the file
display(img)# Shows image


# In[18]:


import numpy as np
img=np.resize(img, (28,28))# The inputted image reshaped to 28 * 28
im2arr = np.array(img)# Converts to an array
im2arr = im2arr.reshape(1,28,28)# 1 image 


# In[19]:


y_pred = model.predict(im2arr)
print(y_pred)# Prints values 


# In[20]:


y_hat = np.argmax(y_pred)# Finds prediction
y_hat# Show hand written digit prediction

