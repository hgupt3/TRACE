from os import path, remove
from sys import exit
from mediapipe import solutions
from cv2 import VideoCapture,VideoWriter,VideoWriter_fourcc,cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from moviepy.editor import VideoFileClip, CompositeVideoClip

# Video to be used placed in Clips folderab
videoFile = './Videos/Clips/Clip3.mp4'

# Ratios of the crop width, height, and offsets
# If centered is 1, program ignores offset and centers frame
crop1_x = 50/100
crop1_x_offset = 0/100
crop1_x_centered = 1

crop1_y = 33/100
crop1_y_offset = 0/100
crop1_y_centered = 0

crop2_x = 83/100
crop2_x_offset = 0/100
crop2_x_centered = 1

crop2_y = 60/100
crop2_y_offset = 40/100
crop2_y_centered = 0

# Error checking for all inputs
flag = 0
if not path.exists(videoFile):
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
video = VideoCapture(videoFile)
width = video.get(3)
height = video.get(4)

# Calculations for pixels used in both crops
crop1_x = int(width*crop1_x)
crop1_y = int(height*crop1_y)
if crop1_x_centered:
    crop1_x_offset = int((width-crop1_x)/2)
else:
    crop1_x_offset = int(width*crop1_x_offset)
if crop1_y_centered:
    crop1_y_offset = int((height-crop1_y)/2)
else:
    crop1_y_offset = int(height*crop1_y_offset)

crop2_x = int(width*crop2_x)
crop2_y = int(height*crop2_y)
if crop2_x_centered:
    crop2_x_offset = int((width-crop2_x)/2)
else:
    crop2_x_offset = int(width*crop2_x_offset)
if crop2_y_centered:
    crop2_y_offset = int((height-crop2_y)/2)
else:
    crop2_y_offset = int(height*crop2_y_offset)

# Defining where to write cropped videos and in what format
fourcc = VideoWriter_fourcc(*'mp4v')
crop1 = VideoWriter('./Videos/PostClips/Video1.mp4',fourcc,25.0,(crop1_x,crop1_y))
crop2 = VideoWriter('./Videos/PostClips/Video2.mp4',fourcc,25.0,(crop2_x,crop2_y))

# Player pose decleration 
pose1 = solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
pose2 = solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) 

while video.isOpened():
    
    # Mapping of Player 1
    ret, frame1 = video.read()
    if frame1 is None:
        break
    frame1 = frame1[crop1_y_offset:crop1_y+crop1_y_offset,crop1_x_offset:crop1_x+crop1_x_offset]
    frame1 = cvtColor(frame1, COLOR_BGR2RGB)
    results1 = pose1.process(frame1)
    frame1 = cvtColor(frame1, COLOR_RGB2BGR)
    solutions.drawing_utils.draw_landmarks(frame1, results1.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    crop1.write(frame1)
    crop1.write(frame1)
    
    # Mapping of Player 2
    ret, frame2 = video.read()
    if frame2 is None:
        break
    frame2 = frame2[crop2_y_offset:crop2_y+crop2_y_offset,crop2_x_offset:crop2_x+crop2_x_offset]
    frame2 = cvtColor(frame2, COLOR_BGR2RGB)
    results2 = pose2.process(frame2)
    frame2 = cvtColor(frame2, COLOR_RGB2BGR)
    solutions.drawing_utils.draw_landmarks(frame2, results2.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    crop2.write(frame2)
    crop2.write(frame2)
    
video.release()
crop1.release()
crop2.release()

# Combining the seperate clips to make a single video file
clipMain = VideoFileClip(videoFile)
clip1 = VideoFileClip("./Videos/PostClips/Video1.mp4").set_position((crop1_x_offset,crop1_y_offset))
clip2 = VideoFileClip("./Videos/PostClips/Video2.mp4").set_position((crop2_x_offset,crop2_y_offset))
result = CompositeVideoClip([clipMain, clip1, clip2])

# Checking if user wants to delete previous result or create new file
if path.exists("./Videos/PostClips/Result.mp4"):
    i = 1
    while True:
        if not path.exists("./Videos/PostClips/Result"+str(i)+".mp4"):
            written = 1
            result.write_videofile("./Videos/PostClips/Result"+str(i)+".mp4", verbose=False, logger=None)
            print("File successfully created at /Videos/PostClips/Result"+str(i)+".mp4")
            break
        i += 1
else:
    result.write_videofile("./Videos/PostClips/Result.mp4", verbose=False, logger=None)
    print("File successfully created at /Videos/PostClips/Result.mp4")

# Removing temporary files
remove("./Videos/PostClips/Video1.mp4")
remove("./Videos/PostClips/Video2.mp4")