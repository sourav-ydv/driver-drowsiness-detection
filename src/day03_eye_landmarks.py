import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]

def draw_eye_points(frame, landmarks, eye_points, w, h, color):
    for idx in eye_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 4, color, -1)

cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            draw_eye_points(frame, landmarks, LEFT_EYE_POINTS, w, h, (0, 255, 0))
            draw_eye_points(frame, landmarks, RIGHT_EYE_POINTS, w, h, (255, 0, 0))

            cv2.putText(frame, "left eye (green)  right eye (blue)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "6 points per eye for EAR",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("eye landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()