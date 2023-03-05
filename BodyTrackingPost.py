import cv2
import mediapipe as mp
import os

# Scanned both halves of video feed seperately for bodies and combined them for final output
# Mediapipe's pose library maps only one person from the video feed 

cam = cv2.VideoCapture(os.path. join('Videos', 'Clip.mp4'))

pose1 = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # First half player pose declaration
pose2 = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # Second half player pose declaration

while cam.isOpened():
    ret, video = cam.read()
    height, width, channels = video.shape
    
    # Player 1 
    ret, frame1 = cam.read()
    frame1 = cv2.rectangle(frame1, (0, 0), (int(width), int(height/3)), (0, 0, 0), -1) # Blocking right half of video feed 
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    frame1.flags.writeable = False
    results1 = pose1.process(frame1) # Coordinate calculation
    mp.solutions.drawing_utils.draw_landmarks(video, results1.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    
    # Player 2
    ret, frame2 = cam.read()
    frame2 = cv2.rectangle(frame2, (0, int(height/3)), (int(width), int(height)), (0, 0, 0), -1) # Blocking left half of video feed 
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    frame2.flags.writeable = False
    results2 = pose2.process(frame2) # Coordinate calculation
    mp.solutions.drawing_utils.draw_landmarks(video, results2.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    
    cv2.imshow('TRACE', video)
    if cv2.waitKey(40) & 0xFF == ord('q'):  # Exit video feed when 'q' pressed
        break
    
cam.release()
cv2.destroyAllWindows()
