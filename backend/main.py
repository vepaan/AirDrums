import cv2
import numpy as np
import mediapipe as mp

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


cap = cv2.VideoCapture();
best_idx = find_best_cam()

cap.open(best_idx)
if not cap.isOpened():
    raise Exception("Cannot open camera")

print("Using camera ", best_idx)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AirDrums", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
