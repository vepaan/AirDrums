import cv2
import numpy as np
import mediapipe as mp

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


cap = cv2.VideoCapture();
best_idx = find_best_cam()
kalman_filters = init_kalman_filter()

cap.open(best_idx)
if not cap.isOpened():
    raise Exception("Cannot open camera")

print("Using camera ", best_idx)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            #now we apply kalman filter here
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) #pixel covo

                measured = np.array([[np.float32(x)], [np.float32(y)]])
                kalman_filters[i].correct(measured)
                prediction = kalman_filters[i].predict()
                px, py = int(prediction[0]), int(prediction[1])

                #drawing both og and filtered points
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) #red - raw
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1) #green - kalman

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AirDrums", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
