import cv2
import numpy as np
import mediapipe as mp
import threading

#This finds the best camera available and returns the index
def find_best_cam():
    max_res, best_idx = 0, -1
    for i in range(10):
        temp = cv2.VideoCapture(i)
        if temp.isOpened():
            width = temp.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = temp.get(cv2.CAP_PROP_FRAME_HEIGHT)
            res = width * height
            print("Cam ", i, " detected with res: ", width, "x", height)

            if res > max_res:
                res = max_res
                best_idx = i
            
            temp.release()

    if best_idx == -1:
        raise Exception("No cam found")

    return best_idx


#Initialize Kalman filtering to better capture jitters not picked up by mediapipe
def init_kalman_filter():
    filters = []
    num_landmarks = len(mp.solutions.hands.HandLandmark)

    for _ in range(num_landmarks):
        # 4 state variables (x, y, vx, vy) and 2 measurement variables (x, y)
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                        [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                        [0, 1, 0, 1], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]], np.float32) * 0.03
        
        filters.append(kalman)

    return filters



#apply kalman function to actually apply the function
def apply_kalman_filter(frame, landmarks):
    #now we apply kalman filter here
    for i, landmark in enumerate(landmarks.landmark):
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) #pixel covo

        measured = np.array([[np.float32(x)], [np.float32(y)]])
        kalman_filters[i].correct(measured)
        prediction = kalman_filters[i].predict()
        px, py = int(prediction[0]), int(prediction[1])

        #drawing both og and filtered points
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) #red - raw
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1) #green - kalman



#we process the detected landmark results by drawing them onto the screen
def process_detection_results(frame, results, result_type, apply_kalman=False):

    if result_type == 'hand':
        landmarks_list = results.multi_hand_landmarks
        connections = mp_hands.HAND_CONNECTIONS

        if landmarks_list:
            for landmark in landmarks_list:
                if apply_kalman:
                    apply_kalman_filter(frame=frame, landmarks=landmark)

                mp_drawing.draw_landmarks(frame, landmark, connections)

    elif result_type == 'pose':
        landmarks_list = results.pose_landmarks
        connections = mp_pose.POSE_CONNECTIONS

        if landmarks_list:
            if apply_kalman:
                apply_kalman_filter(frame=frame, landmarks=landmarks_list)

            mp_drawing.draw_landmarks(frame, landmarks_list, connections)

    else:
        raise Exception("Unsupported type of detection. Choose either 'hand' or 'pair'")

    

#setup camera start
cap = cv2.VideoCapture();
best_idx = find_best_cam()
kalman_filters = init_kalman_filter()

cap.open(best_idx)
if not cap.isOpened():
    raise Exception("Cannot open camera")

print("Using camera ", best_idx)


#loading and initializing hand and pose detection model
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils


#camera instance
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    #now we process both the results in two different threads for parallelism
    process_detection_results(frame=frame,
                              results=hand_results,
                              result_type='hand',
                              apply_kalman=True)

    cv2.imshow("AirDrums", frame)

    if cv2.waitKey(1) == ord('q'):
        break


#terminating camera task
cap.release()
cv2.destroyAllWindows()
