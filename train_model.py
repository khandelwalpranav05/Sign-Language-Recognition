from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras import optimizers

recognition = Sequential()

recognition.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

recognition.add(MaxPooling2D(pool_size=(2,2)))

recognition.add(Convolution2D(32,(3,3),activation = 'relu'))
recognition.add(MaxPooling2D(pool_size = (2,2)))

recognition.add(Convolution2D(32,(3,3),activation = 'relu'))
recognition.add(MaxPooling2D(pool_size = (2,2)))

recognition.add(Flatten())

recognition.add(Dense(256,activation = 'relu'))
recognition.add(Dropout(0.5))
recognition.add(Dense(26,activation='softmax'))

recognition.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

trainingImageGenerator = ImageDataGenerator(
					rescale = 1./255,
					shear_range = 0.2,
					zoom_range = 0.2,
					horizontal_flip = True)

testImageGenerator = ImageDataGenerator(rescale = 1./255)

training = trainingImageGenerator.flow_from_directory(
					'data/training_set',
					target_size = (64,64),
					batch_size = 32,
					class_mode = 'categorical')

testing = testImageGenerator.flow_from_directory(
					'data/test_set',
					target_size = (64,64),
					batch_size = 32,
					class_mode = 'categorical')

recognition.fit_generator(
			training,
			steps_per_epoch = 800,
			epochs = 8,
			validation_data = testing,
			validation_steps = 6500)

print("Traing ends here")

import h5py
recognition.save('Trained_model.h5')

print("Model Saved")