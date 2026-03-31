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
    sr = 44100
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, False)
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))

beep_mild    = generate_beep(600, 0.3)
beep_warning = generate_beep(1000, 0.5)
beep_danger  = generate_beep(1800, 0.8)

print("loading models...")
eye_model  = tf.keras.models.load_model("models/eye_model_finetuned.h5")
face_model = tf.keras.models.load_model("models/face_model_finetuned.h5")
print("models ready")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

EAR_THRESHOLD   = 0.20
MAR_THRESHOLD   = 0.60
PITCH_THRESHOLD = 30.0
YAW_THRESHOLD   = 35.0

MILD_SECONDS   = 1.5
WARN_SECONDS   = 3.0
DANGER_SECONDS = 5.0
YAWN_LIMIT     = 3

class PERCLOS:
    def __init__(self, window=60, fps=15):
        self.size = window * fps
        self.data = []
        self.open_count = 0
        self.blink_buf = 0
        self.high_start = None
        self.sustain_until = 0
        self.last_beep = 0

    def update(self, closed):
        if closed:
            self.open_count = 0
            self.data.append(1)
        else:
            self.open_count += 1
            self.data.append(0)
            if self.open_count > 75:
                self.data = []
                self.open_count = 0
        if len(self.data) > self.size:
            self.data.pop(0)

    def value(self):
        if not self.data:
            return 0.0
        return sum(self.data) / len(self.data) * 100

    def reset(self):
        self.__init__()

perclos = PERCLOS()

def ear_calc(lm, idx, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A+B)/(2*C), pts

def mar_calc(lm, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in MOUTH]
    C = dist.euclidean(pts[0], pts[1])
    A = dist.euclidean(pts[2], pts[3])
    B = dist.euclidean(pts[4], pts[5])
    D = dist.euclidean(pts[6], pts[7])
    return (A+B+D)/(3*C), pts

def head_pose(lm, w, h):
    pts = np.array([
        (lm[1].x*w, lm[1].y*h),
        (lm[152].x*w, lm[152].y*h),
        (lm[226].x*w, lm[226].y*h),
        (lm[446].x*w, lm[446].y*h),
        (lm[57].x*w, lm[57].y*h),
        (lm[287].x*w, lm[287].y*h)
    ], dtype="double")

    model = np.array([
        (0,0,0),(0,-330,-65),(-225,170,-135),
        (225,170,-135),(-150,-150,-125),(150,-150,-125)
    ])

    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype="double")
    distc = np.zeros((4,1))

    ok, rvec, _ = cv2.solvePnP(model, pts, cam, distc)
    if not ok:
        return 0,0

    rmat,_ = cv2.Rodrigues(rvec)
    ang, *_ = cv2.RQDecomp3x3(rmat)
    pitch,yaw = ang[0], ang[1]

    if pitch < -90: pitch += 180
    elif pitch > 90: pitch -= 180

    return pitch,yaw

def crop_eye(frame,lm,idx,w,h,pad=15):
    pts=[(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
    x1=max(0,min(p[0] for p in pts)-pad)
    y1=max(0,min(p[1] for p in pts)-pad)
    x2=min(w,max(p[0] for p in pts)+pad)
    y2=min(h,max(p[1] for p in pts)+pad)
    return frame[y1:y2,x1:x2]

def eye_pred(img):
    if img is None or img.size==0: return 0.5
    try:
        x=cv2.resize(img,(64,64))
        x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)/255.0
        x=np.expand_dims(x,0)
        return eye_model.predict(x,verbose=0)[0][0]
    except: return 0.5

def face_pred(frame,lm,w,h):
    try:
        xs=[int(lm[i].x*w) for i in range(0,468,10)]
        ys=[int(lm[i].y*h) for i in range(0,468,10)]
        x1=max(0,min(xs)-20); y1=max(0,min(ys)-20)
        x2=min(w,max(xs)+20); y2=min(h,max(ys)+20)
        f=frame[y1:y2,x1:x2]
        if f.size==0: return 0.5
        f=cv2.resize(f,(96,96))
        f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)/255.0
        f=np.expand_dims(f,0)
        return face_model.predict(f,verbose=0)[0][0]
    except: return 0.5

def level_fn(d):
    if d<MILD_SECONDS: return 0,"alert",(0,255,0),None
    if d<WARN_SECONDS: return 1,"mild",(0,255,255),beep_mild
    if d<DANGER_SECONDS: return 2,"warning",(0,165,255),beep_warning
    return 3,"danger",(0,0,255),beep_danger

cap=cv2.VideoCapture(1)

eyes_start=None
closed_dur=0
blinks=0
yawns=0
yawn_start=None
face_buf=[]
head_start=None

print("system running")

with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True) as mesh:

    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break

        frame=cv2.flip(frame,1)
        h,w=frame.shape[:2]

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=mesh.process(rgb)

        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark

            ear,lp=ear_calc(lm,LEFT_EYE,w,h)
            ear2,rp=ear_calc(lm,RIGHT_EYE,w,h)
            ear=(ear+ear2)/2

            e=(eye_pred(crop_eye(frame,lm,LEFT_EYE,w,h))+
               eye_pred(crop_eye(frame,lm,RIGHT_EYE,w,h)))/2

            f_raw=face_pred(frame,lm,w,h)
            face_buf.append(f_raw)
            if len(face_buf)>8: face_buf.pop(0)
            f=sum(face_buf)/len(face_buf)

            mar,_=mar_calc(lm,w,h)

            if mar>MAR_THRESHOLD:
                if yawn_start is None: yawn_start=time.time()
                if time.time()-yawn_start>=1.5:
                    y_status="yawning"
                else:
                    y_status="opening"
            else:
                if yawn_start and time.time()-yawn_start>=1.5:
                    yawns+=1
                yawn_start=None
                y_status="normal"

            pitch,yaw=head_pose(lm,w,h)

            if pitch>PITCH_THRESHOLD:
                p_status="drooping"
                if head_start is None: head_start=time.time()
            elif abs(yaw)>YAW_THRESHOLD:
                p_status="away"; head_start=None
            else:
                p_status="normal"; head_start=None

            head_dur=time.time()-head_start if head_start else 0

            ear_v=1 if ear<EAR_THRESHOLD else 0
            eye_v=1 if e>0.5 else 0
            face_v=1 if f>0.5 else 0
            votes=ear_v+eye_v+face_v

            closed=votes>=2

            perclos.blink_buf = perclos.blink_buf+1 if closed else 0
            perclos.update(perclos.blink_buf>=4)
            p_val=perclos.value()

            if closed:
                if eyes_start is None: eyes_start=time.time()
                closed_dur=time.time()-eyes_start
            else:
                if eyes_start and closed_dur<MILD_SECONDS:
                    blinks+=1
                eyes_start=None; closed_dur=0

            lvl,status,color,beep=level_fn(closed_dur)

            if head_dur>=3 and lvl<2:
                lvl,status,color,beep=2,"head drooping",(0,165,255),beep_warning

            if yawns>=YAWN_LIMIT and lvl<1:
                lvl,status,color,beep=1,"fatigue",(0,255,255),beep_mild

            if y_status=="yawning" and lvl<1:
                lvl,status,color,beep=1,"yawning",(0,255,255),beep_mild

            now=time.time()
            if p_val>15 and face_v==1:
                if perclos.high_start is None:
                    perclos.high_start=now
                if now-perclos.high_start>=2:
                    perclos.sustain_until=now+5
            else:
                perclos.high_start=None

            if now<perclos.sustain_until and lvl<2:
                lvl,status,color,beep=2,"high perclos",(0,165,255),beep_warning

            if beep and not pygame.mixer.get_busy():
                if now-perclos.last_beep>=1:
                    beep.play()
                    perclos.last_beep=now

            cv2.putText(frame,f"ear {ear:.3f}",(10,30),0,0.7,color,2)
            cv2.putText(frame,f"perclos {p_val:.1f}",(10,60),0,0.7,color,2)
            cv2.putText(frame,f"votes {votes}",(10,90),0,0.7,color,2)
            cv2.putText(frame,status,(10,120),0,1,color,2)

        cv2.imshow("driver system",frame)

        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('r'):
            perclos.reset()
            face_buf.clear()
            blinks=0;yawns=0
            yawn_start=None
            eyes_start=None
            closed_dur=0
            head_start=None

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("\nsummary")
print("blinks:",blinks)
print("yawns:",yawns)
print("perclos:",perclos.value())