import mediapipe as mp
import cv2

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)  # You can replace 0 with the path to a video file if needed

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(rgb_frame)

    # Extract landmark points
    if results.pose_landmarks:
        # Extract body pose landmarks
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z  # x, y, and z coordinates of the landmark
            print(f"Landmark {i}: x={x}, y={y}, z={z}")
            # Do something with the landmark coordinates (e.g., draw points on the frame)

    if results.face_landmarks:
        # Extract face landmarks
        for i, landmark in enumerate(results.face_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z  # x, y, and z coordinates of the landmark
            print(f"Landmark {i}: x={x}, y={y}, z={z}")
            # Do something with the landmark coordinates (e.g., draw points on the frame)

    if results.left_hand_landmarks:
        # Extract left hand landmarks
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z  # x, y, and z coordinates of the landmark
            print(f"Landmark {i}: x={x}, y={y}, z={z}")
            # Do something with the landmark coordinates (e.g., draw points on the frame)

    if results.right_hand_landmarks:
        # Extract right hand landmarks
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z  # x, y, and z coordinates of the landmark
            print(f"Landmark {i}: x={x}, y={y}, z={z}")
            # Do something with the landmark coordinates (e.g., draw points on the frame)

    # Render the landmarks on the frame
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Holistic Landmarks', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
