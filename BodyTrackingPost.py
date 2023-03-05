import cv2
import mediapipe as mp
import moviepy.editor as me
import os

from moviepy.editor import VideoFileClip, clips_array, vfx

# Scanned both halves of video feed seperately for bodies and combined them for final output
# Mediapipe's pose library maps only one person from the video feed 

cam = cv2.VideoCapture(os.path.join('Videos','Clips','Clip3.mp4'))

pose1 = mp.solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) # First half player pose declaration
pose2 = mp.solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) # Second half player pose declaration
video1 = cv2.VideoWriter('Video1.avi',-1,25,(1920,1080))
video2 = cv2.VideoWriter('Video2.avi',-1,25,(1920,1080))


while cam.isOpened():
    ret, video = cam.read()
    height, width, channels = video.shape
    
    # Player 1 
    ret, frame1 = cam.read()
    frame1 = frame1[int(height*0):int(height),int(width*0):int(width)]
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results1 = pose1.process(frame1) # Coordinate calculation
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame1, results1.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    video1.write(frame1)
    
    # Player 2
    ret, frame2 = cam.read()
    frame2 = frame2[int(height*0):int(height),int(width*0):int(width)]
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results2 = pose2.process(frame2) # Coordinate calculation
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame2, results2.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    video2.write(frame2)
    
    # clipMain = VideoFileClip(video)
    # clip1 = VideoFileClip(frame1)
    # clip2 = VideoFileClip(frame2)
    # final = me.CompositeVideoClip([clipMain, clip1.set_position((10,10)), clip2.set_position((500,500))])
    
    cv2.imshow('TRACE', video)
    if cv2.waitKey(40) & 0xFF == ord('q'):  # Exit video feed when 'q' pressed
        break
    
cam.release()
video1.release()
video2.release()
cv2.destroyAllWindows()