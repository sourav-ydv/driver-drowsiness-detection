import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))

    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])

    ear = (A + B) / (2.0 * C)
    return ear, points

cap = cv2.VideoCapture(1)
print("watching eyes, press q to quit")

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

            left_ear, left_pts = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            for p in left_pts:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)
            for p in right_pts:
                cv2.circle(frame, p, 3, (255, 0, 0), -1)

            cv2.polylines(frame, [np.array(left_pts)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_pts)], True, (255, 0, 0), 1)

            if avg_ear > 0.25:
                color = (0, 255, 0)
                state = "open"
            elif avg_ear > 0.18:
                color = (0, 255, 255)
                state = "closing"
            else:
                color = (0, 0, 255)
                state = "closed"

            cv2.putText(frame, f"ear: {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"state: {state}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"l: {left_ear:.3f}  r: {right_ear:.3f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.putText(frame, "no face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("ear", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()