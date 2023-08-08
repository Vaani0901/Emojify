import cv2
import os
import numpy as np
from keras.models import load_model


emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}

emoji_dict = {0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 2: "./emojis/fearful.png", 3: "./emojis/happy.png", 4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surpriced.png"}


def recognize_emoji(image_path):
    # Load the deep learning model
    model = load_model('model.h5')

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 48x48 pixels
    resized = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)

    # Normalize the pixel values to be between 0 and 1
    normalized = resized / 255.0

    # Reshape the image to have a single channel
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    # Make a prediction using the model
    prediction = model.predict(reshaped)

    # Get the index of the highest predicted emotion
    emotion_index = np.argmax(prediction)

    # Return the corresponding emoji
    return emotion_index
    


#emotion_index = recognize_emoji('C:\\Users\\Vaani Goel\\Desktop\\SML Project\\captured_face.jpg')

# while True:
#     # Get the emotion index of the captured face
#     emotion_index = recognize_emoji('captured_face.jpg')

