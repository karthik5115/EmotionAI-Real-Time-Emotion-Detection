import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('model_file_30epochs.h5')

# Open the camera
video = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the Haar Cascade for face detection
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    # Check if any face was detected
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Extract the face region
            sub_face_img = gray[y:y+h, x:x+w]
            
            # Resize the face image
            resized = cv2.resize(sub_face_img, (48, 48))
            
            # Normalize the resized image
            normalize = resized / 255.0
            
            # Reshape the normalized image
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            
            # Predict the emotion
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
