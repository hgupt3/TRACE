from mediapipe import solutions
from cv2 import VideoCapture,VideoWriter,imshow,cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, circle, waitKey
from TraceHeader import calculatePixels, videoFile

def bodyMap(video):
    width = video.get(3)
    height = video.get(4)
    feetPoints = []
    
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
    
    # Calculations for pixels used in both crops
    crop1 = calculatePixels(crop1, width, height)
    crop2 = calculatePixels(crop2, width, height)
    
    # Player pose decleration 
    mp_pose = solutions.pose
    pose1 = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    pose2 = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.25, min_tracking_confidence=0.25) 
    
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break
            
        # Mapping of Player 1
        frame1 = frame[crop1.yoffset:crop1.y+crop1.yoffset,crop1.xoffset:crop1.x+crop1.xoffset]
        frame1 = cvtColor(frame1, COLOR_BGR2RGB)
        results1 = pose1.process(frame1)
        frame1 = cvtColor(frame1, COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
        
        l1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        l1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        r1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        r1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        # Mapping of Player 2
        frame2 = frame[crop2.yoffset:crop2.y+crop2.yoffset,crop2.xoffset:crop2.x+crop2.xoffset]
        frame2 = cvtColor(frame2, COLOR_BGR2RGB)
        results2 = pose2.process(frame2)
        frame2 = cvtColor(frame2, COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(frame2, results2.pose_landmarks,solutions.pose.POSE_CONNECTIONS)

        l2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        l2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        r2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        r2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        feetPoints.append([[l1_foot_x,l1_foot_y],[r1_foot_x,r1_foot_y],[l2_foot_x,l2_foot_y],[r2_foot_x,r2_foot_y]])
    
    video.release()
    return feetPoints
        