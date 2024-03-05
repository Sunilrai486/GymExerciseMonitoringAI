import mediapipe as mp
import cv2

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#video_path = 'D:/Downloads/fit3d_train/train/s03/videos/65906101/dumbbell_biceps_curls.mp4'
video_path = 'D:/Downloads/fit3d_train/train/s03/videos/58860488/deadlift.mp4'
# Initialize VideoCapture
cap = cv2.VideoCapture(video_path)  # You can replace 0 with the path to a video file if needed

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Extract landmark points
    if results.pose_landmarks:
        print("Body Pose Landmarks:")
        leftshoulderX = 0
        rightshoulderX = 0
        leftshoulderY = 0
        rightshoulderY = 0
        print(results.pose_landmarks)
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z
            #print(f"Landmark {i}: x={x}, y={y}, z={z}")
            if i == 11:
                leftshoulderX = x
                leftshoulderY = y
            if i == 12:
                rightshoulderX = x
                rightshoulderY = y
            if i == 32:
                frame_index_33_x = (leftshoulderX - rightshoulderX)/ 2 + rightshoulderX
                frame_index_33_y = (leftshoulderY + rightshoulderY)/ 2
                print(f"Landmark 33: x={frame_index_33_x}, y={frame_index_33_y}")
    # Render the landmarks on the frame
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pose Landmarks', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
