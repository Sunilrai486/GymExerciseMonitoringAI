import mediapipe as mp
import cv2
import math

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = math.degrees(radians)
    return angle + 360 if angle < 0 else angle

# Function to detect back posture during exercise
def detect_back_posture(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect poses
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Get landmarks of interest (e.g., shoulders and hips)
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
        right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
        # Calculate angle between upper body and hips
        back_angle = calculate_angle(left_shoulder, left_hip, right_hip) + calculate_angle(right_shoulder, right_hip, left_hip)
        # Determine back posture based on angle
        if abs(back_angle - 180) < 20:  # Adjusted threshold for fully straight back
            posture = "Fully Straight"
        else:
            posture = "Not Fully Straight"
        # Overlay text on frame to visualize posture
        cv2.putText(frame, f"Back Posture: {posture} ({back_angle:.2f} degrees)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

video_path = 'D:/Downloads/fit3d_train/train/s03/videos/58860488/deadlift.mp4'

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect back posture during exercise in the frame
    frame_with_posture = detect_back_posture(frame)
    
    # Display the frame
    cv2.imshow('Back Posture During Exercise', frame_with_posture)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
