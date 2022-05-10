import numpy as np
from PIL import Image
import tensorflow as tf

from keras.layers import Activation, Conv2D, MaxPooling2D,UpSampling2D, Input
from keras.layers import BatchNormalization as BN
from tensorflow.keras.optimizers import Adam
from keras.models import Model


!wget https://www.dropbox.com/s/qii26wuxcbxw169/images.npy
!wget https://www.dropbox.com/s/gstpamhin8rlhlg/masks.npy


images=np.load("images.npy")
print(images.shape)
## Crop 512
images=images[:,0:512,0:512,:]

#im=Image.fromarray(images[0])
#im.show()

masks=np.load("masks.npy")
print(masks.shape)
## Crop 512
masks=masks[:,0:512,0:512]

#im=Image.fromarray(masks[0])
#im.show()

#Normalization
images=images/np.max(images)
masks=masks/np.max(masks)

print(np.min(masks[0]),np.max(masks[0]))


def EncBlock(layer,num,filters):
	l=layer
	for i in range(num):
		l=Conv2D(filters, (3, 3), padding='same')(l)
		l=BN()(l)
		l=Activation('relu')(l)

	l=MaxPooling2D((2,2))(l)
	return l


def DecBlock(layer,num,filters,enclayer=None):
	l=layer

	for i in range(num):
		l=Conv2D(filters, (3, 3), padding='same')(l)
		l=BN()(l)
		l=Activation('relu')(l)

	l=UpSampling2D((2,2))(layer)
	l=Conv2D(filters, (3, 3), padding='same')(l)
	l=BN()(l)
	return l

def EncDec(x0):
	## End
	x1=EncBlock(x0,2,32) # 256
	x2=EncBlock(x1,2,64) #128
	x3=EncBlock(x2,2,128) # 64
	x4=EncBlock(x3,2,256) #32 
	
	## Dec
	x5=DecBlock(x4,0,128) #64
	x6=DecBlock(x5,2,64) #128
	x7=DecBlock(x6,2,32) #256
	x8=DecBlock(x7,2,1) #512

	return x8


lin=Input((512,512,3))
out=Activation('sigmoid')(EncDec(lin))

opt = Adam(learning_rate=0.001)

model = Model(inputs=lin, outputs=out)
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['mse'])
    
model.summary()

model.fit(images,masks,batch_size=4,epochs=100,verbose=1)

masks=model.predict(images)


#masks=masks*255
#masks=np.uint8(masks)
#im=Image.fromarray(masks[2,:,:,0])
#im.show()
