import cv2
import mediapipe as mp
import os

from moviepy.editor import VideoFileClip, clips_array, vfx, CompositeVideoClip

cam = cv2.VideoCapture(os.path.join('Videos','Clips','Clip3.mp4'))
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
cam.release()

clipMain = VideoFileClip("./Videos/Clips/Clip3.mp4")
clip1 = VideoFileClip("./Videos/PostClips/Video1.mp4")
clip2 = VideoFileClip("./Videos/PostClips/Video2.mp4")
final = CompositeVideoClip([clipMain, clip1.set_position((frame1_x_offset,frame1_y_offset)), clip2.set_position((frame2_x_offset,frame2_y_offset))])
final.write_videofile("./Videos/PostClips/Final.mp4")