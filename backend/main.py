import cv2
import numpy as np

image = cv2.imread("test.png")
cv2.imshow("AirDrums", image)
cv2.waitKey(0)
cv2.destroyAllWindows()