import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import time

#Random seed for reproducibility
np.random.seed(0)

#parameters for ini data
picturesTaken = 250
batch_size = 16
nb_epoch = 50
img_width, img_height = 150, 150
validationAmount = 1/5

train_data_dir = 'CameraPics/ModelTraining'
validation_data_dir = 'CameraPics/ModelValidation'

#define amount of classes
classes_amount = len(os.listdir(train_data_dir))

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(
        rescale=1./255,        # normalize pixel values to [0,1]
        shear_range=0.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
horizontal_flip=True) # randomly flip the images

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='categorical')

#Model
model = Sequential()
# Convlutional Layer, 32 filters with a 3x3 Convlutional
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))   #32
#relu for nonlinearities
model.add(Activation('relu'))
#Reducing the complexity of feature maps. 2x2 filter
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))       #32
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))      #64
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

#preventing overfitting
#Flatten into 1D
model.add(Flatten())
#Fully connected layer
model.add(Dense(64))               #64
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(int(classes_amount)))     #Output dimension
model.add(Activation('sigmoid'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))    #only for binary classes

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

# categorical_crossentropy for more that 2 classes. binary_crossentropy otherwise
model.compile(loss='binary_crossentropy',
              optimizer=sgd,              #rmsprop
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=4)

nb_train_samples = picturesTaken*classes_amount
nb_validation_samples = picturesTaken*validationAmount*classes_amount

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples / batch_size,
        callbacks=[early_stop])

# Save Model
model_json = model.to_json()
with open("models/basic_cnn_30_epochs_data.json", "w") as json_file:
    json_file.write(model_json)

#Save Weights
model.save_weights('models/basic_cnn_30_epochs_data.h5')


##### TEST

time_ = time.time()
test_img = load_img('some_pics/640_leo_dicaprio_emma_watson.jpg', target_size=(img_width,img_height))
#test_img = load_img('data/validation/Emma_Watson/pic_294.jpg', target_size=(200,200))
test_img.show()
image_as_array = img_to_array(test_img)
image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
prediction = model.predict(image_as_array)              # for vector output
#prediction = model.predict_classes(image_as_array)      # for classes output
print ("Time:%.4f" %(time.time()-time_))
print prediction
