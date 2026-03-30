import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
import time
import pygame

pygame.mixer.init()

def generate_beep(frequency=1000, duration=0.5):
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))

beep_mild    = generate_beep(600, 0.3)
beep_warning = generate_beep(1000, 0.5)
beep_danger  = generate_beep(1800, 0.8)

print("loading models...")
eye_model  = tf.keras.models.load_model("models/eye_model_finetuned.h5")
face_model = tf.keras.models.load_model("models/face_model_finetuned.h5")
print("models loaded")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD  = 0.20
MILD_SECONDS   = 1.5
WARN_SECONDS   = 3.0
DANGER_SECONDS = 5.0

class PERCLOSCalculator:
    def __init__(self, window_seconds=60, fps=15):
        self.window_size = window_seconds * fps
        self.eye_states = []
        self.consecutive_open = 0

    def update(self, is_closed):
        if is_closed:
            self.consecutive_open = 0
            self.eye_states.append(1)
        else:
            self.consecutive_open += 1
            self.eye_states.append(0)
            if self.consecutive_open > 75:
                self.eye_states = []
                self.consecutive_open = 0

        if len(self.eye_states) > self.window_size:
            self.eye_states.pop(0)

    def get(self):
        if len(self.eye_states) == 0:
            return 0.0
        return sum(self.eye_states) / len(self.eye_states) * 100

    def reset(self):
        self.eye_states = []
        self.consecutive_open = 0

perclos = PERCLOSCalculator()

face_buffer = []
FACE_BUFFER = 8

def calculate_ear(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C), pts

def extract_eye(frame, landmarks, eye_indices, w, h, pad=15):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
    x1 = max(0, min(p[0] for p in pts) - pad)
    y1 = max(0, min(p[1] for p in pts) - pad)
    x2 = min(w, max(p[0] for p in pts) + pad)
    y2 = min(h, max(p[1] for p in pts) + pad)
    return frame[y1:y2, x1:x2]

def eye_pred(img):
    if img is None or img.size == 0:
        return 0.5
    try:
        x = cv2.resize(img, (64, 64))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255.0
        x = np.expand_dims(x, axis=0)
        return eye_model.predict(x, verbose=0)[0][0]
    except:
        return 0.5

def face_pred(frame, landmarks, w, h):
    try:
        xs = [int(landmarks[i].x * w) for i in range(0, 468, 10)]
        ys = [int(landmarks[i].y * h) for i in range(0, 468, 10)]
        x1 = max(0, min(xs) - 20)
        y1 = max(0, min(ys) - 20)
        x2 = min(w, max(xs) + 20)
        y2 = min(h, max(ys) + 20)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 0.5
        face = cv2.resize(face, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) / 255.0
        face = np.expand_dims(face, axis=0)
        return face_model.predict(face, verbose=0)[0][0]
    except:
        return 0.5

def get_level(duration):
    if duration < MILD_SECONDS:
        return 0, "alert", (0, 255, 0), None
    if duration < WARN_SECONDS:
        return 1, "mild", (0, 255, 255), beep_mild
    if duration < DANGER_SECONDS:
        return 2, "warning", (0, 165, 255), beep_warning
    return 3, "danger", (0, 0, 255), beep_danger

cap = cv2.VideoCapture(1)

eyes_closed_start = None
closed_duration = 0.0
blink_count = 0

print("starting system")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_ear, lpts = calculate_ear(lm, LEFT_EYE, w, h)
            right_ear, rpts = calculate_ear(lm, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2

            le = extract_eye(frame, lm, LEFT_EYE, w, h)
            re = extract_eye(frame, lm, RIGHT_EYE, w, h)
            e_pred = (eye_pred(le) + eye_pred(re)) / 2

            f_raw = face_pred(frame, lm, w, h)
            face_buffer.append(f_raw)
            if len(face_buffer) > FACE_BUFFER:
                face_buffer.pop(0)
            f_pred = sum(face_buffer) / len(face_buffer)

            ear_v  = 1 if avg_ear < EAR_THRESHOLD else 0
            eye_v  = 1 if e_pred > 0.5 else 0
            face_v = 1 if f_pred > 0.5 else 0
            votes = ear_v + eye_v + face_v

            closed = votes >= 2

            if not hasattr(perclos, "buf"):
                perclos.buf = 0

            perclos.buf = perclos.buf + 1 if closed else 0
            perclos.update(perclos.buf >= 4)
            p_val = perclos.get()

            if closed:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                closed_duration = time.time() - eyes_closed_start
            else:
                if eyes_closed_start and closed_duration < MILD_SECONDS:
                    blink_count += 1
                eyes_closed_start = None
                closed_duration = 0.0

            lvl, status, color, beep = get_level(closed_duration)

            if p_val > 15 and face_v == 1 and lvl < 2:
                lvl = 2
                status = "high perclos"
                color = (0, 165, 255)
                beep = beep_warning

            if beep and not pygame.mixer.get_busy():
                beep.play()

            cv2.putText(frame, f"ear {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"perclos {p_val:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"votes {votes}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()