import cv2 as cv

cap = cv.VideoCapture(0)

while True:
	ret, img = cap.read()
	img = cv.flip(img, 1)
	cv.imshow("Hola mundo", img)

	k = cv.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv.destroyAllWindows()