# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import LearningRateScheduler as LRS
from keras.preprocessing.image import ImageDataGenerator


batch_size = 32
num_classes = 20
epochs = 150



#### LOAD AND TRANSFORM

## Download: ONLY ONCE!
!wget https://www.dropbox.com/s/kdhn10jwj99xkv7/data.tgz
!tar xvzf data.tgz
#####


# Load 
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Stats
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

## View some images
plt.imshow(x_train[2,:,:,: ] )
plt.show()


## Transforms
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


x_train /= 255
x_test /= 255


## Labels
y_train=y_train-1

y_test=y_test-1

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

###########################################################


## DEF A BLOCK CONV + BN + GN + CONV + BN + GN + MAXPOOL 
def CBGN(model,filters,lname,ishape=0):
  if (ishape!=0):
    model.add(Conv2D(filters, (3, 3), padding='same',
                 input_shape=ishape))
  else:
    model.add(Conv2D(filters, (3, 3), padding='same'))

    
  model.add(BN())
  model.add(GN(0.3))
  model.add(Activation('relu'))

  model.add(Conv2D(filters, (3, 3), padding='same'))
  model.add(BN())
  model.add(GN(0.3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),name=lname))
  
  return model

############################################################

## DEF CNN TOPOLOGY  1
model1 = Sequential()

model1=CBGN(model1,32,'conv_model1_1',x_train.shape[1:])
model1=CBGN(model1,64,'conv_model1_2')
model1=CBGN(model1,128,'conv_model1_3')
model1.add(Dropout(0.5,name='m1'))

## DEF CNN TOPOLOGY  2
model2 = Sequential()
model2=CBGN(model2,32,'conv_model2_1',x_train.shape[1:])
model2=CBGN(model2,64,'conv_model2_2')
model2=CBGN(model2,128,'conv_model2_3')
model2.add(Dropout(0.5,name='m2'))


def outer_product(x):
  phi_I = tf.einsum('ijkm,ijkn->imn',x[0],x[1])		# Einstein Notation  [batch,1,1,depth] x [batch,1,1,depth] -> [batch,depth,depth]
  phi_I = tf.reshape(phi_I,[-1,128*128])	        # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
  phi_I = tf.divide(phi_I,31*31)								  # Divide by feature map size [sizexsize]

  y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))		# Take signed square root of phi_I
  z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)								              # Apply l2 normalization
  return z_l2



conv1=model1.get_layer('m1').output 
conv2=model2.get_layer('m2').output 

x = Lambda(outer_product, name='outer_product')([conv1,conv2])

predictions=Dense(num_classes, activation='softmax', name='predictions')(x)

model = Model(inputs=[model1.input,model2.input], outputs=predictions)
  
model.summary()


##############################################
## DEFINE A DATA AUGMENTATION GENERATOR
## WITH MULTIPLE INPUTS
##############################################

datagen = ImageDataGenerator(
  width_shift_range=0.2,
  height_shift_range=0.2,
  rotation_range=20,
  zoom_range=[1.0,1.2],
  horizontal_flip=True)



def multiple_data_generator(generator, X,Y,bs):
    genX = generator.flow(X, Y,batch_size=bs)
    while True:
      [Xi,Yi] = genX.next()
      yield [Xi,Xi],Yi

##############################################


## TRAINING with DA and LRA
history=model.fit_generator(multiple_data_generator(datagen,x_train, y_train,batch_size),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=epochs,
                            callbacks=[set_lr],
                            verbose=1)

## check how to feed test data for validation 
