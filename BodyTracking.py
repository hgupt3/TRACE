import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cam = cv2.VideoCapture(1)   # 0 for windows, 1 for mac

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cam.isOpened():
        ret, frame = cam.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('TRACE', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
    