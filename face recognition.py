import cv2
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
import nltk
nltk.download('punkt')


# Define emotion labels and their corresponding colors
emotion_colors = {'positive': (0, 255, 0), 'negative': (0, 0, 255), 'neutral': (255, 255, 0)}

# Load the face detector and emotion lexicon
face_cascade = cv2.CascadeClassifier(r"C:\Users\asus\Desktop\haarcascade_frontalface_default.xml")

nltk.download('sentiwordnet')


# Define a function to get the emotion label for a given text
def get_emotion(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())

    # Define emotion labels and their corresponding scores
    emotion_scores = {'positive': 0, 'negative': 0, 'neutral': 0}

    # Loop through each word and determine its sentiment score
    for token in tokens:
        # Get the synset for the current word
        synset = list(swn.senti_synsets(token))
        if synset:
            # Calculate the positive, negative, and objective scores for the synset
            pos_score = synset[0].pos_score()
            neg_score = synset[0].neg_score()
            obj_score = synset[0].obj_score()

            # Determine the overall sentiment score for the current word
            if pos_score > neg_score and pos_score > obj_score:
                emotion_scores['positive'] += 1
            elif neg_score > pos_score and neg_score > obj_score:
                emotion_scores['negative'] += 1
            else:
                emotion_scores['neutral'] += 1

    # Determine the overall emotion label based on the sentiment scores
    if emotion_scores['positive'] > emotion_scores['negative'] and emotion_scores['positive'] > emotion_scores[
        'neutral']:
        return 'positive'
    elif emotion_scores['negative'] > emotion_scores['positive'] and emotion_scores['negative'] > emotion_scores[
        'neutral']:
        return 'negative'
    else:
        return 'neutral'


# Initialize the camera feed
cap = cv2.VideoCapture(0)

# Loop through each frame of the camera feed
while True:
    # Read the current frame from the camera feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face and get its emotion label
    for (x, y, w, h) in faces:
        # Extract the face ROI from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Convert the face ROI to grayscale and resize it to 100x100 pixels
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (100, 100))

        # Convert the face ROI to text and get its emotion label
        text = ""
        try:
            text = pytesseract.image_to_string(face_gray)
        except:
            pass
        emotion = get_emotion(text)

        # Draw a rectangle around the face ROI and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_colors[emotion], 2)
        cv2.putText(frame, emotion, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_colors[emotion], 2)

        # Display the current frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()