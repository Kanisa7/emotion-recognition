
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
from datetime import datetime
import requests
import json

# Load the face detection and emotion classification models
face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\facial recog\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Admin\Desktop\facial recog\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define the directory containing images
image_directory = r'C:\Users\Admin\Desktop\facial recog\images'
results_file = r'C:\Users\Admin\Desktop\facial recog\results.txt'
list=[];
# Open the results file for writing
with open(results_file, 'w') as file:
    # Iterate through all files in the image directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            img_path = os.path.join(image_directory, filename)
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            # Process each detected face
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]

                    # Write the results to the file
                    file.write(f'Image: {filename}, Emotion: {label}\n')
                    list.append(filename)
                    print(f'Image: {filename}, Emotion: {label}\n')

                    # Draw the face and label on the image
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    file.write(f'Image: {filename}, Emotion: No Faces Detected\n')
        
        params = {
            'img':'hhhh'
                }
        try:
            response = requests.get("https://example.com/api/endpoint", params=params)
            #response.raise_for_status()  # Raise an exception for bad responses
            #api_data = response.json()   # Assuming API returns JSON data with image URLs or base64 encoded images
            print(api_data)
    
        except:
               print("failed")   	        

            # Display the image with detected faces and emotions
        cv2.imshow('Emotion Detector', frame)
        cv2.waitKey(100)  # Display each image for 1 second
 
# Release resources
cv2.destroyAllWindows()
