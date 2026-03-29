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

beep_mild    = generate_beep(600, 0.3)
beep_warning = generate_beep(1000, 0.5)
beep_danger  = generate_beep(1800, 0.8)

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20

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

def get_alert_level(duration):
    if duration < 1.5:
        return 0, "awake", (0, 255, 0), None
    if duration < 3.0:
        return 1, "mild drowsy", (0, 255, 255), beep_mild
    if duration < 5.0:
        return 2, "warning", (0, 165, 255), beep_warning
    return 3, "danger", (0, 0, 255), beep_danger

def draw_bar(frame, w, h, duration):
    max_d = 5.0
    width = int((min(duration, max_d) / max_d) * (w - 20))

    cv2.rectangle(frame, (10, h - 40), (w - 10, h - 15), (50, 50, 50), -1)

    if width > 0:
        if duration < 1.5:
            color = (0, 255, 0)
        elif duration < 3.0:
            color = (0, 255, 255)
        elif duration < 5.0:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (10, h - 40), (10 + width, h - 15), color, -1)

    cv2.putText(frame, "drowsiness", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cap = cv2.VideoCapture(1)

eyes_closed_start = None
closed_duration = 0.0
blink_count = 0

print("graduated alert system started")

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

            if avg_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                closed_duration = time.time() - eyes_closed_start
            else:
                if eyes_closed_start is not None:
                    if closed_duration < 1.5:
                        blink_count += 1
                eyes_closed_start = None
                closed_duration = 0.0

            level, status, color, beep = get_alert_level(closed_duration)

            if level > 0:
                overlay = frame.copy()
                alpha = 0.15 * level
                cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            if beep and not pygame.mixer.get_busy():
                beep.play()

            cv2.polylines(frame, [np.array(left_pts)], True, color, 1)
            cv2.polylines(frame, [np.array(right_pts)], True, color, 1)

            cv2.putText(frame, f"ear: {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"blinks: {blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if closed_duration > 0:
                cv2.putText(frame, f"closed: {closed_duration:.1f}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, status, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            responses = ["", "gentle alert", "slow down", "pull over"]
            if level > 0:
                cv2.putText(frame, responses[level], (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            draw_bar(frame, w, h, closed_duration)

        cv2.imshow("driver system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()