
import face_recognition
import os
import cv2
import numpy as np

# Path to known faces
known_faces_dir = 'known_faces'
known_faces_encodings = []
known_faces_names = []

# Load and encode known faces
for name in os.listdir(known_faces_dir):
    if not name.startswith('.'):
        for filename in os.listdir(f"{known_faces_dir}/{name}"):
            if filename.endswith(('jpg', 'jpeg', 'png')):
                image = face_recognition.load_image_file(f"{known_faces_dir}/{name}/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_faces_encodings.append(encoding)
                known_faces_names.append(name)
def recognize_faces_in_image(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Recognized Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
recognize_faces_in_image('test_image.jpg')
def recognize_faces_in_video(video_source=0):
    video_capture = cv2.VideoCapture(video_source)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
recognize_faces_in_video()

