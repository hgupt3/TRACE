import cv2
import mediapipe as mp
import os

from moviepy.editor import VideoFileClip, clips_array, vfx, CompositeVideoClip

# Scanned both halves of video feed seperately for bodies and combined them for final output
# Mediapipe's pose library maps only one person from the video feed 

cam = cv2.VideoCapture(os.path.join('Videos','Clips','Clip3.mp4'))

pose1 = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25) # First half player pose declaration
pose2 = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25) # Second half player pose declaration

ret, video = cam.read()
height, width, channels = video.shape

frame1_x = int(width/2)
frame1_x_offset = int(width/4)
frame1_y = int(height/3)
frame1_y_offset = 0

frame2_x = int(width*10/12)
frame2_x_offset = int(width*1/12)
frame2_y = int(height*3/5)
frame2_y_offset = int(height*2/5)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video1 = cv2.VideoWriter('./Videos/PostClips/Video1.mp4',fourcc,25.0,(frame1_x,frame1_y))
video2 = cv2.VideoWriter('./Videos/PostClips/Video2.mp4',fourcc,25.0,(frame2_x,frame2_y))


while cam.isOpened():
    # Player 1 
    ret, frame1 = cam.read()
    frame1 = frame1[frame1_y_offset:frame1_y+frame1_y_offset,frame1_x_offset:frame1_x+frame1_x_offset]
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results1 = pose1.process(frame1) # Coordinate calculation
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame1, results1.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    video1.write(frame1)
    video1.write(frame1)
    
    # Player 2
    ret, frame2 = cam.read()
    frame2 = frame2[frame2_y_offset:frame2_y+frame2_y_offset,frame2_x_offset:frame2_x+frame2_x_offset]
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results2 = pose2.process(frame2) # Coordinate calculation
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame2, results2.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    video2.write(frame2)
    video2.write(frame2)
    
cam.release()
video1.release()
video2.release()