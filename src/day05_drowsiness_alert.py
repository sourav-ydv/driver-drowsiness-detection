import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import pygame

pygame.mixer.init()

def generate_beep(frequency=1000, duration=0.5):
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    wave_stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(wave_stereo)

beep_sound = generate_beep(1000, 0.5)

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
ALERT_SECONDS = 2.0

def calculate_ear(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C), points

def draw_eye_outline(frame, points, color):
    cv2.polylines(frame, [np.array(points)], True, color, 1)
    for p in points:
        cv2.circle(frame, p, 3, color, -1)

cap = cv2.VideoCapture(1)

eyes_closed_start = None
alert_active = False
blink_count = 0
closed_duration = 0.0

print("drowsiness detection started, press q to quit")

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

        status_text = "no face"
        status_color = (128, 128, 128)
        avg_ear = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear, left_pts = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                closed_duration = time.time() - eyes_closed_start

                if closed_duration >= ALERT_SECONDS:
                    alert_active = True
                    status_text = f"drowsy ({closed_duration:.1f}s)"
                    status_color = (0, 0, 255)

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

                    if not pygame.mixer.get_busy():
                        beep_sound.play()
                else:
                    status_text = f"closing {closed_duration:.1f}s"
                    status_color = (0, 165, 255)

                eye_color = (0, 0, 255)

            else:
                if eyes_closed_start is not None:
                    if closed_duration < ALERT_SECONDS:
                        blink_count += 1
                    eyes_closed_start = None
                    closed_duration = 0.0
                    alert_active = False

                status_text = "awake"
                status_color = (0, 255, 0)
                eye_color = (0, 255, 0)

            draw_eye_outline(frame, left_pts, eye_color)
            draw_eye_outline(frame, right_pts, eye_color)

            cv2.putText(frame, f"ear: {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if avg_ear >= EAR_THRESHOLD else (0, 0, 255), 2)

            cv2.putText(frame, f"blinks: {blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cv2.imshow("drowsiness detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()