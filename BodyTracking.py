from os import path, remove
from sys import exit
from dataclasses import dataclass
from mediapipe import solutions
from cv2 import VideoCapture,VideoWriter,VideoWriter_fourcc,cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from moviepy.editor import VideoFileClip, CompositeVideoClip
from TraceHeader import checkBounds, checkPath, calculatePixels, videoFile

# Ratios of the crop width, height, and offsets
# If centered is 1, program ignores offset and centers frame
class crop1:
    x: float = 50/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 33/100
    yoffset: float = 0/100
    ycenter: int = 0
    
class crop2:
    x: float = 83/100
    xoffset: float = 0/100
    xcenter: int = 1 
    
    y: float = 60/100
    yoffset: float = 40/100
    ycenter: int = 0

# Error checking for all inputs
checkBounds(crop1, crop2)
checkPath(videoFile)

# Taking video and finding pixel width and height
video = VideoCapture(videoFile)
width = video.get(3)
height = video.get(4)

# Calculations for pixels used in both crops
crop1 = calculatePixels(crop1, width, height)
crop2 = calculatePixels(crop2, width, height)

# Defining where to write cropped videos and in what format
fourcc = VideoWriter_fourcc(*'mp4v')
clip1 = VideoWriter('./Videos/Results/Video1.mp4',fourcc,25.0,(crop1.x,crop1.y))
clip2 = VideoWriter('./Videos/Results/Video2.mp4',fourcc,25.0,(crop2.x,crop2.y))

# Player pose decleration 
pose1 = solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
pose2 = solutions.pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) 

while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    # Mapping of Player 1
    frame1 = frame[crop1.yoffset:crop1.y+crop1.yoffset,crop1.xoffset:crop1.x+crop1.xoffset]
    frame1 = cvtColor(frame1, COLOR_BGR2RGB)
    results1 = pose1.process(frame1)
    frame1 = cvtColor(frame1, COLOR_RGB2BGR)
    solutions.drawing_utils.draw_landmarks(frame1, results1.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    clip1.write(frame1)
    
    # Mapping of Player 2
    frame2 = frame[crop2.yoffset:crop2.y+crop2.yoffset,crop2.xoffset:crop2.x+crop2.xoffset]
    frame2 = cvtColor(frame2, COLOR_BGR2RGB)
    results2 = pose2.process(frame2)
    frame2 = cvtColor(frame2, COLOR_RGB2BGR)
    solutions.drawing_utils.draw_landmarks(frame2, results2.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    clip2.write(frame2)
    
video.release()
clip1.release()
clip2.release()

# Combining the seperate clips to make a single video file
clipMain = VideoFileClip(videoFile)
clip1 = VideoFileClip("./Videos/Results/Video1.mp4").set_position((crop1.xoffset,crop1.yoffset))
clip2 = VideoFileClip("./Videos/Results/Video2.mp4").set_position((crop2.xoffset,crop2.yoffset))
result = CompositeVideoClip([clipMain, clip1, clip2])

# Allows multiple result files
if path.exists("./Videos/Results/Result.mp4"):
    i = 1
    while True:
        if not path.exists("./Videos/Results/Result"+str(i)+".mp4"):
            written = 1
            result.write_videofile("./Videos/Results/Result"+str(i)+".mp4", verbose=False, logger=None)
            print("File successfully created at /Videos/Results/Result"+str(i)+".mp4")
            break
        i += 1
else:
    result.write_videofile("./Videos/Results/Result.mp4", verbose=False, logger=None)
    print("File successfully created at /Videos/Results/Result.mp4")

# Removing temporary files
remove("./Videos/Results/Video1.mp4")
remove("./Videos/Results/Video2.mp4")