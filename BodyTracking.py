from mediapipe import solutions
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR

mp_pose = solutions.pose

def bodyMap(frame, pose1, pose2, crop1, crop2):
        
    # Mapping of Player 1
    frame1 = frame[crop1.yoffset:crop1.y+crop1.yoffset,crop1.xoffset:crop1.x+crop1.xoffset]
    frame1 = cvtColor(frame1, COLOR_BGR2RGB)
    results1 = pose1.process(frame1)
    frame1 = cvtColor(frame1, COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks,solutions.pose.POSE_CONNECTIONS)
    if results1.pose_landmarks is not None:
        l1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        l1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        r1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop1.x) + crop1.xoffset
        r1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop1.y) + crop1.yoffset

        l1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop1.x) + crop1.xoffset
        l1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop1.y) + crop1.yoffset

        r1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop1.x) + crop1.xoffset
        r1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop1.y) + crop1.yoffset

        nose1_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop1.x) + crop1.xoffset
        nose1_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop1.y) + crop1.yoffset
    else:
        l1_foot_x = None
        l1_foot_y = None

        r1_foot_x = None
        r1_foot_y = None

        l1_hand_x = None
        l1_hand_y = None

        r1_hand_x = None
        r1_hand_y = None

        nose1_x = None
        nose1_y = None

    # Mapping of Player 2
    frame2 = frame[crop2.yoffset:crop2.y+crop2.yoffset,crop2.xoffset:crop2.x+crop2.xoffset]
    frame2 = cvtColor(frame2, COLOR_BGR2RGB)
    results2 = pose2.process(frame2)
    frame2 = cvtColor(frame2, COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(frame2, results2.pose_landmarks,solutions.pose.POSE_CONNECTIONS)

    if results2.pose_landmarks is not None:
        l2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        l2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        r2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop2.x) + crop2.xoffset
        r2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop2.y) + crop2.yoffset

        l2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop2.x) + crop2.xoffset
        l2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop2.y) + crop2.yoffset

        r2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop2.x) + crop2.xoffset
        r2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop2.y) + crop2.yoffset

        nose2_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop2.x) + crop2.xoffset
        nose2_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop2.y) + crop2.yoffset
    else:
        l2_foot_x = None
        l2_foot_y = None

        r2_foot_x = None
        r2_foot_y = None

        l2_hand_x = None
        l2_hand_y = None

        r2_hand_x = None
        r2_hand_y = None

        nose2_x = None
        nose2_y = None

    return ([[[l1_foot_x,l1_foot_y],[r1_foot_x,r1_foot_y],[l2_foot_x,l2_foot_y],[r2_foot_x,r2_foot_y]], [[l1_hand_x,l1_hand_y],[r1_hand_x,r1_hand_y],[l2_hand_x,l2_hand_y],[r2_hand_x,r2_hand_y]], [[nose1_x, nose1_y], [nose2_x, nose2_y]]])