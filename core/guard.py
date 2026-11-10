"""공개 배포용 안전장치: 비밀번호, AI 사용 한도, 녹음 길이 제한."""
from __future__ import annotations

import hmac
import threading
import time
from datetime import date

import streamlit as st

from core import settings

DAILY_AI_LIMIT = settings.get_int("DAILY_AI_LIMIT", 60)      # 앱 전체 하루 AI 요청 수
SESSION_AI_LIMIT = settings.get_int("SESSION_AI_LIMIT", 30)  # 브라우저 한 번 열었을 때 AI 요청 수
MAX_RECORD_SEC = settings.get_int("MAX_RECORD_SEC", 180)     # 녹음 최대 길이(초)
MAX_LOGIN_TRIES = 5
LOCK_SECONDS = 300


# ---------------------------------------------------------------------------
# 비밀번호
# ---------------------------------------------------------------------------
def password_enabled() -> bool:
    return not settings.is_placeholder(settings.get("APP_PASSWORD"))


def require_password() -> None:
    """APP_PASSWORD 가 설정돼 있으면 맞는 비밀번호를 넣기 전까지 앱을 멈춥니다."""
    if not password_enabled() or st.session_state.get("authed"):
        return
    ss = st.session_state
    ss.setdefault("login_fail", 0)
    ss.setdefault("locked_until", 0.0)

    st.markdown("## AI 면접 연습기")
    st.write("비공개로 운영하는 연습 도구예요. 비밀번호를 입력해 주세요.")
    if time.time() < ss.locked_until:
        left = int(ss.locked_until - time.time())
        st.error(f"비밀번호를 여러 번 틀렸어요. {left // 60}분 {left % 60}초 뒤에 다시 시도하세요.")
        st.stop()

    with st.form("login"):
        pw = st.text_input("비밀번호", type="password")
        ok = st.form_submit_button("들어가기", type="primary")
    if ok:
        if hmac.compare_digest(pw.encode(), settings.get("APP_PASSWORD").encode()):
            ss.authed = True
            ss.login_fail = 0
            st.rerun()
        ss.login_fail += 1
        if ss.login_fail >= MAX_LOGIN_TRIES:
            ss.locked_until = time.time() + LOCK_SECONDS
            ss.login_fail = 0
            st.error("비밀번호를 5번 틀려서 5분 동안 잠겼어요.")
        else:
            st.error(f"비밀번호가 맞지 않아요. ({ss.login_fail}/{MAX_LOGIN_TRIES})")
    st.stop()


# ---------------------------------------------------------------------------
# AI 사용 한도
# ---------------------------------------------------------------------------
@st.cache_resource
def _global_counter() -> dict:
    # 앱이 실행되는 동안 모든 사용자가 함께 쓰는 카운터 (앱이 다시 시작되면 0부터)
    return {"day": date.today().isoformat(), "count": 0, "lock": threading.Lock()}


def usage() -> dict:
    g = _global_counter()
    with g["lock"]:
        if g["day"] != date.today().isoformat():
            g["day"], g["count"] = date.today().isoformat(), 0
        today = g["count"]
    mine = st.session_state.get("ai_used", 0)
    return {"today": today, "today_left": max(0, DAILY_AI_LIMIT - today),
            "mine": mine, "mine_left": max(0, SESSION_AI_LIMIT - mine)}


def can_use_ai() -> tuple[bool, str]:
    u = usage()
    if u["today_left"] <= 0:
        return False, f"오늘 AI 사용 한도({DAILY_AI_LIMIT}회)를 다 썼어요. 내일 다시 이용해 주세요."
    if u["mine_left"] <= 0:
        return False, f"이번 접속에서 쓸 수 있는 AI 요청({SESSION_AI_LIMIT}회)을 다 썼어요."
    return True, ""


def count_ai_use() -> None:
    g = _global_counter()
    with g["lock"]:
        if g["day"] != date.today().isoformat():
            g["day"], g["count"] = date.today().isoformat(), 0
        g["count"] += 1
    st.session_state["ai_used"] = st.session_state.get("ai_used", 0) + 1
