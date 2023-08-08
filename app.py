import cv2
import streamlit as st
import numpy as np
import prediction 

# Define a function to detect faces in real-time
def detect_faces():

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Set up a counter to keep track of the number of faces detected
    count = 0

    # Set up a flag to indicate whether face detection is in progress
    detect_flag = True

    # Set up a progress bar to show the progress of face detection
    progress_bar = st.progress(0)

    # Set up a placeholder for displaying the captured images
    captured_images_placeholder = st.empty()

    # Set up a placeholder for displaying the avatar image
    avatar_image_placeholder = st.empty()

    # Define the columns for displaying the images
    col1, col2 = st.columns(2)

    # Loop through frames
    while detect_flag:

        # Read a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        # Draw a rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Capture the face and store it
            face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            cv2.imwrite('captured_face.jpg', face_img)

            # Update the counter
            count = 1

            # Display the captured images in the first column
            col1.image(face_img, caption=f"Face Capture", use_column_width=True)
            
            # Call the recognize_emoji function to get the emotion index
            emotion_index = prediction.recognize_emoji('C:\\Users\\Vaani Goel\\Desktop\\SML Project\\captured_face.jpg')

            # Display the resulting avatar image in the second column
            avatar = cv2.imread(prediction.emoji_dict[emotion_index])
            col2.image(avatar, caption= prediction.emotion_dict[emotion_index], use_column_width=True)

            # Stop face detection if image is captured
            if count == 1:
                detect_flag = False
                break

        # Stop the loop if the 'q' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()


# Set page title and favicon
st.set_page_config(page_title="Live Face Detection Web App", page_icon=":random:")


# Define CSS styles
st.markdown("""
<style>
    .title {
        font-size: 48px;
        color: #ffffff;
        background-color: #0078d7;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .button {
        font-size
        color: #ffffff;
        background-color: #2ecc71;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
    }
    .output {
        margin-top: 20px;
        padding: 20px;
        border: 1px solid #cccccc
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Define the title of the web page
st.markdown('<h1 class="title">Emojify Web App</h1>', unsafe_allow_html=True)

st.subheader("Real-time Emotion Based Avatar for Autism Assistance")

# Add a button to start face detection
if st.button("Start Face Detection"):

    # Show a message to indicate that face detection is in progress
    st.info("Face detection is in progress... Please wait.")

    # Call the function to detect faces
    detect_faces()

    # Show a message to indicate that face detection is complete
    st.success("Face detection is complete!")


# Add a footer
st.markdown("---")
st.markdown("Created by The Emojify Team")    








