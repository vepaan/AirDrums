import cv2

image = cv2.imread('test.png')
cv2.imshow("AirDrums", image)
cv2.waitKey(0)
#test
cv2.destroyAllWindows()
