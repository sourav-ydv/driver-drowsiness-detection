import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("cannot open webcam")
    exit()

print("webcam started, press q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("cannot read frame")
        break

    print(frame.shape)

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "press q to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("webcam closed")