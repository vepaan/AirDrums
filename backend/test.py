import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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


#setup camera start
cap = cv2.VideoCapture();
best_idx = find_best_cam()

cap.open(best_idx)
if not cap.isOpened():
    raise Exception("Cannot open camera")

print("Using camera ", best_idx)


#loading and initializing hand detection model
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    print('hand landmarker result: {}'.format(result))
    global results
    results = result

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands = 2, 
    min_hand_detection_confidence=0.7, 
    min_tracking_confidence=0.3
)

hand_detector = HandLandmarker.create_from_options(options)
results = None #gloval variable representing latest results from detection
frame_timestamp_ms = 0 #initializing frame timestamp

#camera instance
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

    frame_timestamp_ms += int((cap.get(cv2.CAP_PROP_POS_MSEC) % 1) * 1000)

    hand_detector.detect_async(mp_image, frame_timestamp_ms)

    if results and results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            
            for landmark in hand_landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green dots for landmarks

            connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                        (0, 17)]  # Palm

            for connection in connections:
                x1, y1 = int(hand_landmarks[connection[0]].x * frame.shape[1]), int(hand_landmarks[connection[0]].y * frame.shape[0])
                x2, y2 = int(hand_landmarks[connection[1]].x * frame.shape[1]), int(hand_landmarks[connection[1]].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines for connections

    cv2.imshow("AirDrums", frame)

    if cv2.waitKey(1) == ord('q'):
        break


#terminating camera task
cap.release()
cv2.destroyAllWindows()
