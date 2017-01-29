import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, Dropout
import os
import numpy as np
from scipy import misc
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

X=np.load('X_Data.npy')
y=np.load('y_Data.npy')

X=X.astype("float32")

Enc=LabelEncoder()
y= Enc.fit_transform(y)

Class_List=Enc.classes_
with file('Class_list.npy', 'w') as Classfile:
    np.save(Classfile,Class_List)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


model = Sequential() # Define sequential model

# The first convolutional layer takes the input of the colour image, which is 200 x 200 in size, and applies
# 32 different filters (each 2 X 2) in order to detect features in the shoes. The filters used are chosen, like
# the weights in the network, by backpropagation. Each filter is convolved over the input matrix to produce
# another image by means of matrix multiplication. An activation function is then applied element
# wise to the resulting image. The Relu activation function is defined as the max(0, x), in essence eliminating
# negative values.

model.add(Convolution2D(nb_filter = 32, nb_col = 3, nb_row = 3, input_shape = (200, 200, 3), activation = 'relu' ))

# MaxPooling is then applied to the convoluted image. The purpose of MaxPooling is to down sample the image, thus
# reducing the dimentionality of the feature space and allowing for assumptions to be made about features contained
# in the sub-regions binned. A 2 x 2 matrix convolves over the image matrix and takes the maximum value contained
# within that region. The advantaged of MaxPooling is that it prevents the network from over-fitting and also
# reduces the computational time in training.

model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout randomly selects 25% of the input units, setting them to zero. This also prevents over-fitting.
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filter = 32, nb_col = 3, nb_row = 3, activation = 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten, flattens the matrix into an array for input into the network.
model.add(Flatten())

# The initial weights of the nodes in the network are randomly assigned from the Uniform distribution, which are then
# updated in the backpropagation process.
model.add(Dense(128, init = 'uniform', activation = 'relu'))
model.add(Dense(128, init = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))

# Like in most networks, the final layer contains the activation function softmax. The softmax function squashes a
# K-dimentional vector z of arbitrary real values, to a K-dimentional vector of real values between 0 and 1 that
# sum to 1. The output of the softmax function will be the probability distribution over the different classes.
model.add(Dense(8, init = 'uniform', activation = 'softmax')) # 11 classes

# To evaluate the weights in the networks we must specify the loss function, and to optimise the weights we must
# specify the optimiser. Because this is a classification problem, the loss function used will be binary_crossentropy.
# The optimiser adam was chosen for no other reason than efficientsy. The evlauation metric used to judge
# the networks performance will be accuracy

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model
model.fit(X_train, keras.utils.np_utils.to_categorical(y_train), nb_epoch = 20, batch_size = 32, verbose = 5)

model_json=model.to_json()
with open("model.json",'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Evaluate model
score = model.evaluate(X_test, keras.utils.np_utils.to_categorical(y_test))
print score
