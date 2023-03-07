import cv2
import mediapipe as mp
import os.path
from sys import exit

# Scanned both halves of video feed seperately for the two players
# Mediapipe's pose library maps only one person at a time

# Video to be used placed in Clips folder
videoFile = './Videos/Clips/Clip3.mp4'

# Ratios of the crop width, height, and offsets
crop1_x = 50/100
crop1_x_offset = 25/100
crop1_y = 33/100
crop1_y_offset = 0/100

crop2_x = 83/100
crop2_x_offset = 8/100
crop2_y = 60/100
crop2_y_offset = 40/100

# Error checking of all the inputs
flag = 0
if not os.path.exists(videoFile):
    print("videoFile: path "+videoFile+" does not exist")
    flag = 1
if (crop1_x+crop1_x_offset>1):
    print("crop1: x Coordinates out of bounds")
    flag = 1
if (crop1_y+crop1_y_offset>1):
    print("crop1: y Coordinates out of bounds")
    flag = 1
if (crop2_x+crop2_x_offset>1):
    print("crop2: x Coordinates out of bounds")
    flag = 1
if (crop2_y+crop2_y_offset>1):
    print("crop2: y Coordinates out of bounds")
    flag = 1
if (flag):
    exit()

# Taking video and finding pixel width and height
video = cv2.VideoCapture(videoFile)
width = video.get(3)
height = video.get(4)

# Calculations for pixels used in both crops
crop1_x = int(width*crop1_x)
crop1_x_offset = int(width*crop1_x_offset)
crop1_y = int(height*crop1_y)
crop1_y_offset = int(height*crop1_y_offset)

crop2_x = int(width*crop2_x)
crop2_x_offset = int(width*crop2_x_offset)
crop2_y = int(height*crop2_y)
crop2_y_offset = int(height*crop2_y_offset)

# Defining where to write cropped videos and in what format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
crop1 = cv2.VideoWriter('./Videos/PostClips/Video1.mp4',fourcc,25.0,(crop1_x,crop1_y))
crop2 = cv2.VideoWriter('./Videos/PostClips/Video2.mp4',fourcc,25.0,(crop2_x,crop2_y))

# Player pose decleration 
pose1 = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25)
pose2 = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25) 

while video.isOpened():
    
    # Player 1 locatating and mapping
    ret, frame1 = video.read()
    if frame1 is None:
        break
    frame1 = frame1[crop1_y_offset:crop1_y+crop1_y_offset,crop1_x_offset:crop1_x+crop1_x_offset]
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results1 = pose1.process(frame1) # Coordinate calculation
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame1, results1.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    crop1.write(frame1)
    crop1.write(frame1)
    
    # Player 2 locating and mapping
    ret, frame2 = video.read()
    if frame2 is None:
        break
    frame2 = frame2[crop2_y_offset:crop2_y+crop2_y_offset,crop2_x_offset:crop2_x+crop2_x_offset]
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # Mediapipe requires RGB
    results2 = pose2.process(frame2) # Coordinate calculation
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(frame2, results2.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS) # Maps results onto video feed
    crop2.write(frame2)
    crop2.write(frame2)
    
video.release()
crop1.release()
crop2.release()
print("Successfully exported videos to ./Vides/PostClips")