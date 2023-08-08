import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D

#preprocessing training and validation data
train_dir = 'C:\\Users\\Vaani Goel\\Desktop\\SML Project\\SML_data\\train'
val_dir = 'C:\\Users\\Vaani Goel\\Desktop\\SML Project\\SML_data\\test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)  #scales the pixel value from [0,255] to [0,1]


#generator objects for both training and validation sets are created using the 'flow_from_directory' method
#yields batches of augmented and preprocessed images with labels
train_generator = train_datagen.flow_from_directory(      #preprocess taining data
        train_dir,
        target_size=(48,48),      #target_size parameter specifies the size to which the images should be resized
        batch_size=64,            #batch_size is the size of the batches that will be yielded by the generator
        color_mode="grayscale",   # color_mode specifies the number of color channels
        class_mode='categorical') #class_mode specifies the type of labels

validation_generator = val_datagen.flow_from_directory(     #preprocess validation data
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = Sequential()   #The model is created using the Sequential model class from Keras. 
                               #This allows us to add layers to the model in a sequential order.


#input_shape parameter specifies the shape of the input data.
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))) #kernel_size parameter specifies the size of the filters
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  #activation specifies the activation function to be used
emotion_model.add(MaxPooling2D(pool_size=(2, 2))) 
#MaxPooling2D layer downsamples the feature maps produced by the convolutional layer by taking the maximum value in each pooling window
#The pool_size parameter specifies the size of the pooling window.

emotion_model.add(Dropout(0.25))
#Two dropout layers (Dropout) are added after the first and last pooling layers. 
# This layer randomly drops out a certain percentage of the neurons in the layer during training to prevent overfitting.

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# flattened output is passed through two fully connected (Dense) layers.
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# The loss parameter specifies the loss function to be used, which is categorical cross-entropy in this case.
emotion_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001, decay=1e-6), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#The metrics parameter specifies the metric to be used to evaluate the performance of the model during training, which is accuracy in this case.

emotion_model_info = emotion_model.fit(
        #The train_generator and validation_generator are passed as inputs to this method
        train_generator, 
        #The steps_per_epoch and validation_steps parameters specify the number of batches to be yielded by the generators in one epoch.
        steps_per_epoch=28709 // 64,
        epochs=60, 
        validation_data=validation_generator,  
        validation_steps=7178 // 64)

emotion_model.save('model.h5')  

from sklearn.metrics import confusion_matrix

# get the true labels and predicted labels for the validation set 
Y_true = validation_generator.classes
Y_pred = emotion_model.predict(validation_generator)

# get the indices of the class with the highest probability
Y_pred_classes = np.argmax(Y_pred, axis=1)

# calculate the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)


print("Confusion Matrix:\n", confusion_mtx)
