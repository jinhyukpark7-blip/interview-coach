"""카메라 실시간 분석 (MediaPipe Face Landmarker).

- 얼굴 인식, 시선(카메라 응시), 정면 자세, 고개 흔들림, 미소, 눈 깜빡임을 셉니다.
- 영상은 앱이 실행되는 곳(내 PC 또는 Streamlit 서버)에서 실시간으로 계산만 하고 저장하지 않습니다.
  OpenAI 로 보내지 않습니다.
"""
from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

GAZE_KEYS = [f"eyeLook{d}{s}" for d in ("In", "Out", "Up", "Down") for s in ("Left", "Right")]


def ensure_model() -> Path:
    """얼굴 인식 모델 파일이 없으면 처음 한 번 내려받습니다(약 3.7MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


@dataclass
class FaceReport:
    seconds: float = 0.0
    face_ratio: float = 0.0          # 얼굴이 화면에 잡힌 비율
    eye_contact_ratio: float = 0.0   # 카메라를 바라본 비율
    frontal_ratio: float = 0.0       # 고개가 정면인 비율
    smile_ratio: float = 0.0         # 미소 띤 비율
    blinks_per_min: float = 0.0
    head_movement: float = 0.0       # 고개 흔들림(작을수록 안정)
    face_score: int = 0
    notes: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.monotonic()
        self.frames = 0
        self.face = 0
        self.eye = 0
        self.frontal = 0
        self.smile = 0
        self.blinks = 0
        self._eye_closed = False
        self.nose = []
        self.live = {"face": False, "eye": False, "frontal": False, "smile": False}


class FaceCoach:
    """streamlit-webrtc 의 video_frame_callback 에 넘기는 처리기."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()
        self._stats = _Stats()
        self._last_ts = 0

    # --- 영상 프레임마다 호출 -------------------------------------------------
    def process(self, frame):
        import av
        import cv2

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 거울처럼 보이게
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)

        h, w = img.shape[:2]
        live = {"face": False, "eye": False, "frontal": False, "smile": False}
        with self._lock:
            s = self._stats
            s.frames += 1
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
                live["face"] = True
                s.face += 1

                xs = np.array([p.x for p in lm])
                ys = np.array([p.y for p in lm])
                left, right = sorted((lm[234].x, lm[454].x))
                face_w = max(right - left, 1e-6)
                yaw = (lm[1].x - left) / face_w
                pitch = (lm[1].y - lm[10].y) / max(lm[152].y - lm[10].y, 1e-6)
                frontal = 0.36 < yaw < 0.64 and 0.40 < pitch < 0.70
                gaze_ok = max((bs.get(k, 0.0) for k in GAZE_KEYS), default=0.0) < 0.45
                eye = frontal and gaze_ok
                smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2 > 0.35
                closed = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2 > 0.5
                if closed and not s._eye_closed:
                    s.blinks += 1
                s._eye_closed = closed

                s.frontal += frontal
                s.eye += eye
                s.smile += smile
                s.nose.append((lm[1].x / face_w, lm[1].y / face_w))
                if len(s.nose) > 3000:
                    s.nose = s.nose[-3000:]
                live.update(frontal=frontal, eye=eye, smile=smile)

                # 얼굴 테두리: 카메라를 보면 초록, 아니면 주황 (글자는 넣지 않음)
                x1, y1 = int(xs.min() * w), int(ys.min() * h)
                x2, y2 = int(xs.max() * w), int(ys.max() * h)
                color = (90, 190, 60) if eye else (40, 140, 240)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            s.live = live
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # --- 화면에서 읽어 가는 값 -----------------------------------------------
    def live_state(self) -> dict:
        with self._lock:
            s = self._stats
            return dict(s.live, seconds=time.monotonic() - s.t0,
                        eye_ratio=(s.eye / s.face) if s.face else 0.0)

    def reset(self):
        with self._lock:
            self._stats.reset()

    def report(self) -> FaceReport:
        with self._lock:
            s = self._stats
            secs = time.monotonic() - s.t0
            rep = FaceReport(seconds=round(secs, 1))
            if s.frames == 0:
                rep.notes = ["카메라 영상이 들어오지 않았어요."]
                return rep
            rep.face_ratio = round(s.face / s.frames, 2)
            if s.face:
                rep.eye_contact_ratio = round(s.eye / s.face, 2)
                rep.frontal_ratio = round(s.frontal / s.face, 2)
                rep.smile_ratio = round(s.smile / s.face, 2)
            rep.blinks_per_min = round(s.blinks / max(secs / 60, 1 / 60), 1)
            if len(s.nose) > 5:
                rep.head_movement = round(float(np.std(np.array(s.nose), axis=0).mean()) * 100, 1)
        rep.face_score, rep.notes = _face_score(rep)
        return rep


def _face_score(r: FaceReport) -> tuple[int, list]:
    notes = []
    smile_part = 100 if 0.08 <= r.smile_ratio <= 0.6 else (60 if r.smile_ratio < 0.08 else 75)
    steady = max(0.0, 100 - max(0.0, r.head_movement - 2.5) * 12)
    blink = 100 if r.blinks_per_min <= 30 else max(50.0, 100 - (r.blinks_per_min - 30) * 2.5)
    score = (0.40 * r.eye_contact_ratio * 100 + 0.20 * r.frontal_ratio * 100
             + 0.15 * steady + 0.15 * smile_part + 0.10 * blink) * min(1.0, r.face_ratio / 0.8)

    if r.face_ratio < 0.8:
        notes.append("얼굴이 화면 밖으로 자주 나갔어요. 카메라 높이와 거리를 맞춰 주세요.")
    if r.eye_contact_ratio < 0.6:
        notes.append("카메라를 바라본 시간이 짧아요. 면접관 눈 대신 렌즈를 보며 말해 보세요.")
    if r.head_movement > 4:
        notes.append("고개 움직임이 많아요. 어깨를 펴고 고개를 고정한 채 말해 보세요.")
    if r.smile_ratio < 0.08:
        notes.append("표정이 굳어 보여요. 답변 시작과 끝에 가볍게 미소를 지어 보세요.")
    if r.blinks_per_min > 35:
        notes.append("눈 깜빡임이 많아요. 긴장했을 때 나타나는 경우가 많으니 숨을 천천히 쉬어 보세요.")
    if not notes:
        notes.append("시선과 자세가 안정적이에요.")
    return int(round(score)), notes
