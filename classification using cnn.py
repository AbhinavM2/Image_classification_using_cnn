#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as mat
import os
import cv2
import numpy as np
import tqdm as tq


# In[17]:


labels={}
m=0
Cat = "C:\\Users\\Asus\\Documents\\PetImages\\Cat"
Dog = "C:\\Users\\Asus\\Documents\\PetImages\\Dog"
train_x=[]
train_y=[]
labels[Cat]=1;
labels[Dog]=0;
for label in labels:
    for i in tq.tqdm(os.listdir(label)):
        try:
            m=m+1
            path=os.path.join(label,i)
            img=cv2.imread(path,cv2.IMREAD_COLOR)
            img=cv2.resize(img,(64,64))
            img=np.array(img)
            #print(np.shape(img))
            train_x.append(img)
            train_y.append(labels[label])
        except:
            pass
        


# In[19]:


#train_x=np.reshape(train_x,(24946,64,64,3))
print(np.shape(train_x))


# In[13]:


train_y=np.array(train_y)
train_y=np.reshape(train_y,(24946,1))
train_y=train_y.reshape(-1,)


# In[14]:


train_x=train_x/255


# In[15]:


s=np.arange(m)
train_x=train_x[s]
train_y=train_y[s]


# In[10]:


model=keras.Sequential([
                            keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)),
                            keras.layers.MaxPooling2D((2,2)),
                            keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                            keras.layers.MaxPooling2D((2,2)),
                            keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                            keras.layers.MaxPooling2D((2,2)),
                            keras.layers.Flatten(),
                            keras.layers.Dense(64,activation='relu'),
                            keras.layers.Dense(32,activation='relu'),
                            keras.layers.Dense(1,activation='sigmoid')
                        ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[11]:


model.fit(train_x,train_y,batch_size=128,epochs=5)


# In[ ]:




