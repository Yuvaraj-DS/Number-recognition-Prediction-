import cv2

cap = cv2.VideoCapture(0)

# width = int(cap.get(CAP_PROP_WIDTH))
# height = int(cap.get(CAP_PROP_HEIGHT))

while True:

	ret, frame = cap.read()

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.circle(frame, (100, 400), 70, (0, 0, 255), 15)

	cv2.imshow('Application', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()