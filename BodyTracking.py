import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cam = cv2.VideoCapture(1)

pose1 = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
pose2 = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
while cam.isOpened():
    
    ret, video = cam.read()
    video = cv2.flip(video, 1)
    height, width, channels = video.shape
    
    ret, frame1 = cam.read()
    frame1 = cv2.flip(frame1, 1)
    frame1 = cv2.rectangle(frame1, (int(width/2), 0), (int(width), int(height)), (0, 0, 0), -1)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame1.flags.writeable = False
    results1 = pose1.process(frame1)
    
    ret, frame2 = cam.read()
    frame2 = cv2.flip(frame2, 1)
    frame2 = cv2.rectangle(frame2, (0, 0), (int(width/2), int(height)), (0, 0, 0), -1)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame2.flags.writeable = False
    results2 = pose2.process(frame2)
    
    mp_drawing.draw_landmarks(video, results1.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(video, results2.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('TRACE', video)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()