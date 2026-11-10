"""음성 분석: 말하기 속도, 습관어, 목소리 톤, 자신감.

외부 무거운 라이브러리 없이 numpy + scipy 만 사용합니다.
수치는 '연습용 참고 지표'이며 전문 음성 분석 장비의 측정값이 아닙니다.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# 기준값 (필요하면 여기서 조정하세요)
# ---------------------------------------------------------------------------
RATE_RANGE = {
    # (너무 느림, 적당 시작, 적당 끝, 너무 빠름)
    "ko": (200, 250, 330, 380),   # 음절/분
    "en": (90, 120, 160, 190),    # 단어/분
}
RATE_UNIT = {"ko": "음절/분", "en": "단어/분"}

# 간투사(의미 없는 소리)와 말버릇(뜻은 있지만 습관적으로 반복되는 말)
KO_FILLERS = ["음", "으음", "어", "어어", "아", "에", "으", "흠"]
KO_FILLERS_WHEN_TRAILING = ["그", "저", "뭐"]          # "그...", "저," 처럼 끊길 때만
KO_HABITS = ["약간", "이제", "막", "뭔가", "뭐랄까", "사실", "진짜", "그러니까", "그니까", "솔직히"]

EN_FILLERS = ["um", "umm", "uh", "uhh", "er", "erm", "ah", "ahh", "hmm", "mm"]
EN_FILLERS_WHEN_TRAILING = ["a", "the", "so"]          # "a...", "a," 처럼 끊길 때만
EN_HABITS = ["like", "you know", "i mean", "basically", "actually", "kind of", "sort of", "literally"]


@dataclass
class AudioReport:
    duration_sec: float = 0.0
    speaking_sec: float = 0.0
    language: str = "ko"
    speech_rate: float = 0.0
    speech_rate_unit: str = "음절/분"
    speech_rate_label: str = "-"
    speech_rate_score: int = 0
    fillers: dict = field(default_factory=dict)       # 간투사
    habits: dict = field(default_factory=dict)        # 말버릇
    filler_total: int = 0
    filler_per_min: float = 0.0
    filler_score: int = 0
    pause_count: int = 0
    longest_pause_sec: float = 0.0
    pause_ratio: float = 0.0
    pitch_median_hz: float = 0.0
    pitch_variation_st: float = 0.0
    tone_label: str = "-"
    volume_dbfs: float = 0.0
    volume_label: str = "-"
    volume_stability_db: float = 0.0
    confidence_score: int = 0
    confidence_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 공통 도우미
# ---------------------------------------------------------------------------
def _band_score(x: float, bad_lo: float, good_lo: float, good_hi: float, bad_hi: float) -> float:
    """적정 구간이면 100, 벗어날수록 0에 가까워지는 점수."""
    if good_lo <= x <= good_hi:
        return 100.0
    if x < good_lo:
        if x <= bad_lo:
            return max(0.0, 40.0 * (x / bad_lo)) if bad_lo > 0 else 0.0
        return 40.0 + 60.0 * (x - bad_lo) / (good_lo - bad_lo)
    if x >= bad_hi:
        return 40.0
    return 100.0 - 60.0 * (x - good_hi) / (bad_hi - good_hi)


def load_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(io.BytesIO(wav_bytes))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32), int(sr)


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _pitch_track(frames: np.ndarray, sr: int, fmin: float = 75, fmax: float = 400) -> np.ndarray:
    """자기상관 방식의 간단한 기본 주파수 추정. 무성 구간은 0."""
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    win = np.hanning(frames.shape[1]).astype(np.float32)
    out = np.zeros(len(frames), dtype=np.float32)
    nfft = 1 << (2 * frames.shape[1] - 1).bit_length()
    spec = np.fft.rfft(frames * win, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:, : frames.shape[1]]
    for i, a in enumerate(ac):
        if a[0] <= 1e-8 or lag_max >= len(a):
            continue
        seg = a[lag_min:lag_max]
        k = int(np.argmax(seg))
        if seg[k] / a[0] > 0.35:
            out[i] = sr / (lag_min + k)
    return out


# ---------------------------------------------------------------------------
# 신호 분석
# ---------------------------------------------------------------------------
def analyze_signal(x: np.ndarray, sr: int) -> dict:
    duration = len(x) / sr if sr else 0.0
    hop = int(0.010 * sr)
    rms_frames = _frames(x, int(0.025 * sr), hop)
    rms = np.sqrt(np.mean(rms_frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)

    noise = np.percentile(rms, 10)
    thr = max(noise * 3.0, np.max(rms) * 0.08, 1e-3)
    speech = rms > thr
    # 짧은 끊김(0.15초 미만)은 말하는 중으로 메움
    speech = _fill_short_gaps(speech, int(0.15 / 0.010))

    res = {"duration": duration, "speaking": 0.0, "pause_count": 0, "longest_pause": 0.0,
           "pause_ratio": 0.0, "pitch_median": 0.0, "pitch_st": 0.0,
           "volume_db": -90.0, "volume_std": 0.0}
    if not speech.any():
        return res

    first, last = np.argmax(speech), len(speech) - 1 - np.argmax(speech[::-1])
    span = speech[first: last + 1]
    res["speaking"] = (last - first + 1) * 0.010

    # 쉼: 말 사이 0.6초 이상 조용한 구간
    pauses, run = [], 0
    for v in span:
        if not v:
            run += 1
        else:
            if run * 0.010 >= 0.6:
                pauses.append(run * 0.010)
            run = 0
    res["pause_count"] = len(pauses)
    res["longest_pause"] = max(pauses) if pauses else 0.0
    res["pause_ratio"] = float((~span).sum() / len(span))

    res["volume_db"] = float(np.mean(db[speech]))
    res["volume_std"] = float(np.std(db[speech]))

    pitch_frames = _frames(x, int(0.040 * sr), hop)
    n = min(len(pitch_frames), len(speech))
    f0 = _pitch_track(pitch_frames[:n][speech[:n]], sr)
    f0 = f0[f0 > 0]
    if len(f0) > 10:
        med = float(np.median(f0))
        st = 12 * np.log2(f0 / med)
        st = st[np.abs(st) < 12]          # 옥타브 오류 제거
        res["pitch_median"] = med
        res["pitch_st"] = float(np.std(st)) if len(st) else 0.0
    return res


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# 텍스트 분석 (말하기 속도, 습관어)
# ---------------------------------------------------------------------------
def count_units(text: str, lang: str) -> int:
    if lang == "ko":
        return len(re.findall(r"[가-힣]", text))
    return len(re.findall(r"[A-Za-z']+", text))


def count_fillers(text: str, lang: str) -> tuple[dict, dict]:
    t = text.lower()
    fillers, habits = {}, {}
    plain = KO_FILLERS if lang == "ko" else EN_FILLERS
    trailing = KO_FILLERS_WHEN_TRAILING if lang == "ko" else EN_FILLERS_WHEN_TRAILING
    habit_words = KO_HABITS if lang == "ko" else EN_HABITS
    # 영어 질문이라도 한국어 간투사가 섞일 수 있어 함께 셉니다.
    if lang == "en":
        plain = plain + KO_FILLERS

    boundary = r"(?<![가-힣A-Za-z'])"
    end = r"(?![가-힣A-Za-z'])"
    for w in plain:
        c = len(re.findall(boundary + re.escape(w) + end, t))
        if c:
            fillers[w] = c
    for w in trailing:
        # "그...", "저,", "a..." 처럼 바로 뒤에 쉼표·말줄임이 오거나 같은 말이 반복될 때
        pat = boundary + re.escape(w) + r"(?:\s*(?:,|\.\.\.|…|-)|\s+" + re.escape(w) + end + ")"
        c = len(re.findall(pat, t))
        if c:
            label = f"{w}…"
            fillers[label] = fillers.get(label, 0) + c
    for w in habit_words:
        c = len(re.findall(boundary + re.escape(w) + end, t))
        if c:
            habits[w] = c
    return fillers, habits


# ---------------------------------------------------------------------------
# 종합
# ---------------------------------------------------------------------------
def analyze_answer(wav_bytes: bytes, transcript: str, lang: str = "ko") -> AudioReport:
    x, sr = load_wav(wav_bytes)
    sig = analyze_signal(x, sr)
    rep = AudioReport(language=lang, speech_rate_unit=RATE_UNIT[lang])
    rep.duration_sec = round(sig["duration"], 1)
    rep.speaking_sec = round(sig["speaking"], 1)

    minutes = max(sig["speaking"], 1.0) / 60
    units = count_units(transcript, lang)
    rep.speech_rate = round(units / minutes, 0)
    lo_bad, lo, hi, hi_bad = RATE_RANGE[lang]
    if units == 0:
        rep.speech_rate_label = "말소리를 찾지 못함"
    elif rep.speech_rate < lo:
        rep.speech_rate_label = "느림"
    elif rep.speech_rate > hi:
        rep.speech_rate_label = "빠름"
    else:
        rep.speech_rate_label = "적당함"
    rep.speech_rate_score = int(round(_band_score(rep.speech_rate, lo_bad, lo, hi, hi_bad))) if units else 0

    rep.fillers, rep.habits = count_fillers(transcript, lang)
    rep.filler_total = sum(rep.fillers.values()) + sum(rep.habits.values())
    rep.filler_per_min = round(rep.filler_total / minutes, 1)
    rep.filler_score = int(round(max(0.0, 100 - max(0.0, rep.filler_per_min - 1.5) * 12)))

    rep.pause_count = sig["pause_count"]
    rep.longest_pause_sec = round(sig["longest_pause"], 1)
    rep.pause_ratio = round(sig["pause_ratio"], 2)

    rep.pitch_median_hz = round(sig["pitch_median"], 0)
    rep.pitch_variation_st = round(sig["pitch_st"], 1)
    if sig["pitch_median"] == 0:
        rep.tone_label = "측정 불가"
    elif rep.pitch_variation_st < 1.2:
        rep.tone_label = "단조로움"
    elif rep.pitch_variation_st > 5.5:
        rep.tone_label = "높낮이가 큼(긴장했을 수 있음)"
    else:
        rep.tone_label = "자연스러운 억양"

    rep.volume_dbfs = round(sig["volume_db"], 1)
    rep.volume_stability_db = round(sig["volume_std"], 1)
    if rep.volume_dbfs < -35:
        rep.volume_label = "작음"
    elif rep.volume_dbfs > -10:
        rep.volume_label = "큼"
    else:
        rep.volume_label = "적당함"

    rep.confidence_score, rep.confidence_notes = _confidence(rep)
    return rep


def _confidence(r: AudioReport) -> tuple[int, list]:
    notes = []
    vol = _band_score(r.volume_dbfs, -45, -32, -12, -5)
    tone = _band_score(r.pitch_variation_st, 0.4, 1.2, 5.0, 8.0) if r.pitch_median_hz else 50
    pause = 100 - min(100, max(0, (r.pause_ratio - 0.15) * 250))
    longest = 100 - min(60, max(0, (r.longest_pause_sec - 2.0) * 20))
    steady = 100 - min(50, max(0, (r.volume_stability_db - 8) * 6))

    score = (0.22 * vol + 0.18 * tone + 0.18 * pause + 0.10 * longest + 0.07 * steady
             + 0.13 * r.filler_score + 0.12 * r.speech_rate_score)

    if r.volume_label == "작음":
        notes.append("목소리가 작게 들려요. 마이크와 거리를 유지한 채 한 단계 크게 말해 보세요.")
    if r.tone_label == "단조로움":
        notes.append("억양 변화가 적어요. 핵심 단어에서 목소리를 살짝 올려 강조해 보세요.")
    if r.tone_label.startswith("높낮이가 큼"):
        notes.append("목소리 높낮이 변화가 커요. 숨을 고르고 문장 끝을 내려서 마무리해 보세요.")
    if r.pause_ratio > 0.3:
        notes.append("말 사이 멈춤이 많아요. 답변 뼈대(결론→근거→예시)를 먼저 정해 두면 줄어듭니다.")
    if r.longest_pause_sec > 3:
        notes.append(f"가장 긴 멈춤이 {r.longest_pause_sec}초였어요. 생각할 때는 '잠시 정리해 보겠습니다'처럼 말로 이어 주세요.")
    if r.filler_per_min > 4:
        notes.append("습관어가 자주 나와요. 습관어 대신 짧게 멈추는 연습을 해 보세요.")
    if r.speech_rate_label in ("빠름", "느림"):
        speed = "빠른" if r.speech_rate_label == "빠름" else "느린"
        notes.append(f"말하기 속도가 {speed} 편이에요. 적정 범위는 {RATE_RANGE[r.language][1]}~{RATE_RANGE[r.language][2]} {r.speech_rate_unit}입니다.")
    if not notes:
        notes.append("전달 측면에서 큰 문제는 보이지 않아요. 지금 속도와 크기를 유지하세요.")
    return int(round(score)), notes
