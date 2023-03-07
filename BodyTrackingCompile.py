import cv2
import mediapipe as mp
from os import path
from moviepy.editor import VideoFileClip, CompositeVideoClip

# Video to be used placed in Clips folder
videoFile = 'Clip3.mp4'

# Ratios of the crop width, height, and offsets
crop1_x = 50/100
crop1_x_offset = 25/100
crop1_y = 33/100
crop1_y_offset = 0/100

crop2_x = 83/100
crop2_x_offset = 8/100
crop2_y = 60/100
crop2_y_offset = 40/100

# Taking video and finding pixel width and height
video = cv2.VideoCapture(path.join('Videos','Clips',videoFile))
width = video.get(3)
height = video.get(4)
video.release()

# Calculations for pixels used in both crops
crop1_x = int(width*crop1_x)
crop1_x_offset = int(width*crop1_x_offset)
crop1_y = int(height*crop1_y)
crop1_y_offset = int(height*crop1_y_offset)

crop2_x = int(width*crop2_x)
crop2_x_offset = int(width*crop2_x_offset)
crop2_y = int(height*crop2_y)
crop2_y_offset = int(height*crop2_y_offset)

clipMain = VideoFileClip("./Videos/Clips/Clip3.mp4")
clip1 = VideoFileClip("./Videos/PostClips/Video1.mp4")
clip2 = VideoFileClip("./Videos/PostClips/Video2.mp4")
final = CompositeVideoClip([clipMain, clip1.set_position((crop1_x_offset,crop1_y_offset)), clip2.set_position((crop2_x_offset,crop2_y_offset))])
final.write_videofile("./Videos/PostClips/Final.mp4")