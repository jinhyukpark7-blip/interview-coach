"""AI 면접 연습기 (해외영업 신입 · 동남아) — Streamlit 앱.

실행:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import random
import time

import pandas as pd
import streamlit as st

from core import guard, llm, settings
from core.audio_analysis import analyze_answer, load_wav
from core.questions import DEFAULT_QUESTIONS, DEFAULT_ROLE, answer_language
from core.report import build_markdown, grade, summary_rows, total_score

st.set_page_config(page_title="AI 면접 연습기", layout="wide")

TIMER_SEC = 60
NAVY = "#14254a"

# ---------------------------------------------------------------------------
# 스타일 (남색 · 흰색, 큰 글씨)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-size: 18px; }}
.block-container {{ padding-top: 1.6rem; max-width: 1100px; }}
.q-card {{ background:{NAVY}; color:#fff; border-radius:18px; padding:26px 24px; margin: 6px 0 14px; }}
.q-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }}
.q-badge {{ background:#fff; color:{NAVY}; font-weight:700; border-radius:999px; padding:3px 12px; font-size:15px; }}
.q-count {{ font-size:16px; opacity:.85; }}
.q-text {{ font-size: clamp(24px, 3.2vw, 32px); font-weight:700; line-height:1.45; margin:0; }}
.q-tip {{ border-top:1px solid rgba(255,255,255,.25); margin-top:16px; padding-top:12px; font-size:16px; opacity:.92; }}
.q-tip b {{ margin-right:6px; }}
.timer {{ font-size:52px; font-weight:800; color:{NAVY}; line-height:1; font-variant-numeric:tabular-nums; }}
.timer.low {{ color:#c2410c; }}
.chip {{ display:inline-block; border-radius:999px; padding:4px 12px; margin:3px 4px 3px 0; font-size:15px; font-weight:700; }}
.chip.on {{ background:{NAVY}; color:#fff; }}
.chip.off {{ background:#e8ecf5; color:#5a6683; }}
.score-big {{ font-size:44px; font-weight:800; color:{NAVY}; line-height:1.1; }}
div.stButton > button {{ min-height: 52px; font-size: 17px; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 상태 초기화
# ---------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("questions", [dict(q) for q in DEFAULT_QUESTIONS])
    ss.setdefault("set_name", "기본 질문 20개")
    ss.setdefault("set_choice", "기본 질문 20개")
    ss.setdefault("company", "")
    ss.setdefault("role", DEFAULT_ROLE)
    ss.setdefault("filter", "전체")
    ss.setdefault("order", [])
    ss.setdefault("order_sig", None)
    ss.setdefault("pos", 0)
    ss.setdefault("memos", {})          # 화면에만 보관 (파일 저장 안 함)
    ss.setdefault("results", {})
    ss.setdefault("attempt", {})
    ss.setdefault("timer_end", None)    # 진행 중이면 끝나는 시각
    ss.setdefault("timer_left", TIMER_SEC)
    ss.setdefault("collected", [])
    ss.setdefault("citations", [])
    ss.setdefault("generated", [])
    ss.setdefault("ai_summary", "")


guard.require_password()   # APP_PASSWORD 가 있으면 비밀번호를 넣기 전까지 여기서 멈춤
init_state()
ss = st.session_state


def ai_allowed() -> bool:
    """AI 요청 전에 사용 한도를 확인하고, 넘으면 안내를 띄웁니다."""
    if not llm.has_api_key():
        st.error("OpenAI API 키가 없어 AI 기능을 쓸 수 없어요. README 의 'API 키 넣기'를 확인하세요.")
        return False
    ok, msg = guard.can_use_ai()
    if not ok:
        st.warning(msg)
    return ok


def ice_servers() -> list[dict]:
    """카메라 연결용 서버. Streamlit Cloud 에서 카메라가 안 붙으면 TURN 값을 Secrets 에 넣으세요."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_url = settings.get("TURN_URL")
    if turn_url:
        servers.append({"urls": [u.strip() for u in turn_url.split(",") if u.strip()],
                        "username": settings.get("TURN_USERNAME"),
                        "credential": settings.get("TURN_CREDENTIAL")})
    return servers


def qid(q: dict) -> str:
    return hashlib.md5(f'{q["category"]}|{q["question"]}'.encode()).hexdigest()[:10]


def sync_order():
    sig = (ss.filter, ss.set_name, len(ss.questions))
    if ss.order_sig != sig:
        ss.order = [i for i, q in enumerate(ss.questions) if ss.filter == "전체" or q["category"] == ss.filter]
        ss.order_sig = sig
        ss.pos = 0
    if ss.order:
        ss.pos %= len(ss.order)


def use_question_set(questions: list[dict], name: str):
    """콜백(on_click/on_change) 안에서만 부르세요. 위젯 값(filter, set_choice)을 바꾸기 때문입니다."""
    ss.questions = questions
    ss.set_name = name
    ss.set_choice = name
    ss.filter = "전체"
    ss.order_sig = None
    reset_timer()


def on_set_change():
    name = ss.set_choice
    src = DEFAULT_QUESTIONS if name == "기본 질문 20개" else ss.generated
    use_question_set([dict(q) for q in src], name)


# --- 타이머 -----------------------------------------------------------------
def timer_remaining() -> float:
    if ss.timer_end is None:
        return ss.timer_left
    return max(0.0, ss.timer_end - time.time())


def start_timer():
    ss.timer_end = time.time() + (ss.timer_left if ss.timer_left > 0 else TIMER_SEC)
    coach = ss.get("coach")
    if coach is not None and ss.timer_left in (0, TIMER_SEC):
        coach.reset()  # 답변 시작 시점부터 표정·시선을 다시 셈


def pause_timer():
    ss.timer_left = timer_remaining()
    ss.timer_end = None


def reset_timer():
    ss.timer_end = None
    ss.timer_left = TIMER_SEC


def move(step: int):
    if ss.order:
        ss.pos = (ss.pos + step) % len(ss.order)
    reset_timer()


def shuffle():
    random.shuffle(ss.order)
    ss.pos = 0
    reset_timer()


def save_memo(key: str, mid: str):
    ss.memos[mid] = ss.get(key, "")


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## AI 면접 연습기")
    st.caption(f"지원: {ss.company or '회사 미정'} · {ss.role}")
    if llm.has_api_key():
        st.success(f"OpenAI 연결됨 · 모델 {llm.CHAT_MODEL}")
        u = guard.usage()
        st.caption(f"AI 사용: 이번 접속 {u['mine']}/{guard.SESSION_AI_LIMIT}회 · "
                   f"오늘 전체 {u['today']}/{guard.DAILY_AI_LIMIT}회")
    else:
        st.error("OpenAI 키가 없어요. README 의 'API 키 넣기'를 확인하세요.")
    if not guard.password_enabled():
        st.caption("비밀번호가 설정되지 않았어요. 인터넷에 공개할 때는 APP_PASSWORD 를 꼭 설정하세요.")
    elif st.button("로그아웃", use_container_width=True):
        ss.authed = False
        st.rerun()

    st.markdown("### 질문 세트")
    options = ["기본 질문 20개"] + (["맞춤 질문"] if ss.generated else [])
    if ss.set_choice not in options:
        ss.set_choice = options[0]
    st.radio("연습할 질문", options, key="set_choice", on_change=on_set_change, label_visibility="collapsed")

    st.markdown("### 카메라")
    use_camera = st.toggle("카메라로 표정·시선 분석", value=False,
                           help="영상은 앱 서버에서 실시간으로 분석만 하고 저장하지 않아요. OpenAI 로도 보내지 않아요.")


with st.expander("개인정보 안내 · 녹음과 영상은 저장하지 않아요", expanded=not ss.get("notice_seen", False)):
    st.markdown("""
- **카메라 영상**: 앱 서버에서 실시간으로 얼굴 위치·시선만 계산하고, 영상 자체는 저장하지 않아요. OpenAI 로 보내지도 않아요.
- **녹음한 음성**: 받아쓰기와 답변 평가를 위해 **OpenAI 로 전송**돼요. 앱은 음성 파일을 따로 저장하지 않아요.
- **메모와 결과**: 지금 열어 둔 화면에만 있어요. 새로고침하거나 창을 닫으면 사라지니, 필요하면 '종합 리포트'에서 내려받으세요.
- 다른 사람의 얼굴이나 목소리가 들어가지 않게 혼자 있는 곳에서 연습해 주세요.
""")
    if st.button("확인했어요", key="notice_ok"):
        ss.notice_seen = True
        st.rerun()


tab_practice, tab_custom, tab_report = st.tabs(["연습하기", "맞춤 질문 만들기", "종합 리포트"])

# ---------------------------------------------------------------------------
# 1) 연습하기
# ---------------------------------------------------------------------------
def practice_tab():
    st.radio("분류", ["전체", "인성", "직무", "영어"], horizontal=True, key="filter", on_change=reset_timer)
    sync_order()

    if not ss.order:
        st.info("이 분류에는 질문이 없어요. 다른 분류를 골라 주세요.")
        return

    q = ss.questions[ss.order[ss.pos]]
    mid = qid(q)

    st.markdown(f"""
<div class="q-card">
  <div class="q-top"><span class="q-badge">{q['category']} 질문</span>
  <span class="q-count">{ss.pos + 1} / {len(ss.order)} · {ss.set_name}</span></div>
  <p class="q-text">{q['question']}</p>
  <div class="q-tip"><b>답변 포인트</b>{q.get('tip') or '-'}</div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.button("이전", on_click=move, args=(-1,), use_container_width=True)
    c2.button("섞기", on_click=shuffle, use_container_width=True)
    c3.button("다음", on_click=move, args=(1,), use_container_width=True, type="primary")

    left, right = st.columns([1, 1], gap="large")

    # --- 카메라 -------------------------------------------------------------
    with left:
        st.markdown("#### 카메라")
        if not use_camera:
            st.caption("왼쪽 사이드바에서 '카메라로 표정·시선 분석'을 켜면 실제 면접처럼 연습할 수 있어요.")
        else:
            try:
                from streamlit_webrtc import WebRtcMode, webrtc_streamer
                from core.face_analysis import FaceCoach

                if "coach" not in ss:
                    with st.spinner("얼굴 인식 모델을 준비하는 중..."):
                        ss.coach = FaceCoach()
                coach = ss.coach
                webrtc_streamer(
                    key="camera",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=coach.process,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={"iceServers": ice_servers()},
                    async_processing=True,
                    translations={"start": "카메라 켜기", "stop": "카메라 끄기",
                                  "select_device": "장치 선택", "media_api_not_available": "이 브라우저에서는 카메라를 쓸 수 없어요.",
                                  "device_ask_permission": "카메라 사용을 허용해 주세요.",
                                  "device_not_available": "카메라를 찾지 못했어요.",
                                  "device_access_denied": "카메라 권한이 거부됐어요."},
                )
                st.caption("얼굴 테두리가 초록색이면 카메라를 잘 보고 있는 거예요. 주황색이면 시선이 벗어난 상태입니다. "
                           "'카메라 켜기'를 눌러도 화면이 계속 검은색이면 README 의 '카메라가 연결되지 않을 때'를 보세요.")
            except Exception as e:  # noqa: BLE001
                st.error(f"카메라 기능을 시작하지 못했어요: {e}")
                ss.pop("coach", None)

    # --- 타이머 + 실시간 표시 (1초마다 이 부분만 새로고침) ---------------------
    @st.fragment(run_every=1.0)
    def live_panel():
        left_sec = timer_remaining()
        if ss.timer_end is not None and left_sec <= 0:
            ss.timer_end, ss.timer_left = None, 0
        low = " low" if left_sec <= 10 else ""
        st.markdown(f'<div class="timer{low}">{int(round(left_sec))}초</div>', unsafe_allow_html=True)
        st.progress(min(1.0, left_sec / TIMER_SEC))
        b1, b2 = st.columns(2)
        if ss.timer_end is None:
            label = "답변 시작" if ss.timer_left in (0, TIMER_SEC) else "계속"
            b1.button(label, on_click=start_timer, use_container_width=True, type="primary", key="t_start")
        else:
            b1.button("멈춤", on_click=pause_timer, use_container_width=True, key="t_pause")
        b2.button("처음으로", on_click=reset_timer, use_container_width=True, key="t_reset")

        coach = ss.get("coach")
        if use_camera and coach is not None:
            s = coach.live_state()
            chips = [("얼굴 인식", s["face"]), ("카메라 응시", s["eye"]),
                     ("정면 자세", s["frontal"]), ("미소", s["smile"])]
            st.markdown("".join(f'<span class="chip {"on" if on else "off"}">{name}</span>' for name, on in chips)
                        + f'<div style="margin-top:6px;font-size:15px;color:#5a6683">답변 시작 후 카메라 응시 {round(s["eye_ratio"] * 100)}%</div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown("#### 60초 타이머")
        live_panel()

    # --- 녹음 + 메모 ----------------------------------------------------------
    st.markdown("#### 마이크로 답변하기")
    lang = answer_language(q["category"])
    st.caption("영어 질문은 영어로 답해 주세요." if lang == "en" else "실제 면접처럼 소리 내어 답해 주세요. '답변 시작'을 누른 뒤 녹음하면 표정·시선도 같은 구간으로 분석합니다.")
    attempt = ss.attempt.get(mid, 0)
    audio = st.audio_input("녹음 버튼을 누르고 답한 뒤 다시 눌러 멈추세요.", key=f"audio_{mid}_{attempt}")

    memo_key = f"memo_{mid}"
    st.text_area("답변 메모 (저장되지 않아요)", value=ss.memos.get(mid, ""), key=memo_key, height=140,
                 placeholder="핵심 키워드나 답변 흐름을 적어 보세요.", on_change=save_memo, args=(memo_key, mid))

    a1, a2 = st.columns(2)
    analyze_clicked = a1.button("AI 분석 받기", type="primary", use_container_width=True, disabled=audio is None)
    if a2.button("다시 녹음하기", use_container_width=True):
        ss.attempt[mid] = attempt + 1
        reset_timer()
        st.rerun()

    if analyze_clicked and audio is not None:
        if ss.timer_end is not None:
            pause_timer()
        wav = audio.getvalue()
        try:
            x, sr = load_wav(wav)
            rec_sec = len(x) / sr
        except Exception:  # noqa: BLE001
            rec_sec = 0
        if rec_sec > guard.MAX_RECORD_SEC:
            st.error(f"녹음이 {int(rec_sec)}초예요. {guard.MAX_RECORD_SEC}초 이하로 다시 녹음해 주세요.")
        elif ai_allowed():
            with st.status("답변을 분석하는 중...", expanded=True) as status:
                try:
                    guard.count_ai_use()
                    st.write("1/4 음성을 글로 바꾸는 중")
                    transcript = llm.transcribe(wav, lang)
                    st.write("2/4 말하기 속도·습관어·목소리 분석 중")
                    au = analyze_answer(wav, transcript, lang).to_dict()
                    fa = None
                    if use_camera and ss.get("coach") is not None:
                        st.write("3/4 표정·시선 정리 중")
                        fa = ss.coach.report().to_dict()
                    else:
                        st.write("3/4 카메라 꺼짐 — 표정 분석은 건너뜀")
                    st.write("4/4 AI 면접관이 내용을 평가하는 중")
                    delivery = {k: au[k] for k in ("speech_rate", "speech_rate_unit", "speech_rate_label",
                                                   "filler_total", "filler_per_min", "tone_label",
                                                   "volume_label", "pause_count", "longest_pause_sec")}
                    if fa:
                        delivery["카메라 응시 비율"] = fa["eye_contact_ratio"]
                    ai = llm.evaluate_answer(question=q["question"], category=q["category"], tip=q.get("tip", ""),
                                             transcript=transcript, role=ss.role, company=ss.company,
                                             delivery=delivery)
                    ss.results[mid] = {"category": q["category"], "question": q["question"],
                                       "transcript": transcript, "audio": au, "face": fa, "ai": ai,
                                       "total": total_score(ai, au, fa)}
                    status.update(label="분석이 끝났어요.", state="complete", expanded=False)
                except llm.AIError as e:
                    status.update(label="분석하지 못했어요.", state="error")
                    st.error(str(e))
                except Exception as e:  # noqa: BLE001
                    status.update(label="분석하지 못했어요.", state="error")
                    st.error(f"녹음 파일을 처리하지 못했어요: {e}")

    # --- 결과 -----------------------------------------------------------------
    res = ss.results.get(mid)
    if res:
        ai, au, fa = res["ai"], res["audio"], res.get("face")
        st.markdown("---")
        st.markdown("### 이 질문의 결과")
        m = st.columns(4)
        m[0].markdown(f'<div class="score-big">{res["total"]}</div>종합 · {grade(res["total"])}', unsafe_allow_html=True)
        m[1].metric("내용 (AI 평가)", ai.get("overall", 0))
        m[2].metric("목소리 자신감", au["confidence_score"])
        m[3].metric("표정·시선", fa["face_score"] if fa and fa.get("face_ratio") else "-")
        st.markdown(f"**한 줄 총평** {ai.get('summary', '')}")

        t1, t2, t3, t4 = st.tabs(["내용 피드백", "말하기·목소리", "표정·시선", "개선 답변·꼬리 질문"])
        with t1:
            if ai.get("scores"):
                st.bar_chart(pd.DataFrame({"점수": ai["scores"]}), horizontal=True, color=NAVY)
            st.markdown("**논리 흐름**  \n" + ai.get("logic_flow", "-"))
            st.markdown("**잘한 점**\n" + "\n".join(f"- {s}" for s in ai.get("strengths", [])))
            st.markdown("**고칠 점**\n" + "\n".join(f"- {s}" for s in ai.get("improvements", [])))
            if ai.get("english_feedback"):
                st.markdown("**영어 표현 교정**\n" + "\n".join(f"- {s}" for s in ai["english_feedback"]))
            with st.expander("내 답변 받아쓰기 보기"):
                st.write(res["transcript"] or "(말소리를 찾지 못했어요)")
        with t2:
            c = st.columns(3)
            c[0].metric("말하기 속도", f'{au["speech_rate"]:.0f}', help=au["speech_rate_unit"])
            c[0].caption(f'{au["speech_rate_unit"]} · {au["speech_rate_label"]}')
            c[1].metric("습관어", f'{au["filler_total"]}회')
            c[1].caption(f'분당 {au["filler_per_min"]}회')
            c[2].metric("멈춤", f'{au["pause_count"]}회')
            c[2].caption(f'가장 긴 멈춤 {au["longest_pause_sec"]}초')
            counts = {**au["fillers"], **au["habits"]}
            if counts:
                st.markdown("**나온 습관어**")
                st.dataframe(pd.DataFrame([{"습관어": k, "횟수": v,
                                            "종류": "간투사" if k in au["fillers"] else "말버릇"}
                                           for k, v in sorted(counts.items(), key=lambda x: -x[1])]),
                             hide_index=True, use_container_width=True)
            else:
                st.success("습관어가 거의 없었어요.")
            st.markdown(f'**목소리 톤** {au["tone_label"]} (억양 변화 {au["pitch_variation_st"]}반음) · '
                        f'**크기** {au["volume_label"]} · **답변 길이** {au["speaking_sec"]}초')
            st.markdown("**자신감 코칭**\n" + "\n".join(f"- {n}" for n in au["confidence_notes"]))
        with t3:
            if fa and fa.get("face_ratio"):
                c = st.columns(4)
                c[0].metric("카메라 응시", f'{round(fa["eye_contact_ratio"] * 100)}%')
                c[1].metric("정면 자세", f'{round(fa["frontal_ratio"] * 100)}%')
                c[2].metric("미소", f'{round(fa["smile_ratio"] * 100)}%')
                c[3].metric("눈 깜빡임", f'분당 {fa["blinks_per_min"]}회')
                st.markdown("\n".join(f"- {n}" for n in fa.get("notes") or []))
            else:
                st.info("카메라를 켜고 '답변 시작'을 누른 뒤 녹음하면 표정·시선 분석이 나와요.")
        with t4:
            st.markdown("**개선 답변 예시**")
            st.write(ai.get("better_answer", "-"))
            fu = ai.get("follow_up")
            if fu:
                st.markdown(f"**예상 꼬리 질문** {fu}")
                if st.button("꼬리 질문으로 이어서 연습하기", use_container_width=True):
                    new_q = {"category": q["category"], "question": fu,
                             "tip": "앞 답변과 말이 어긋나지 않게, 근거를 하나 더 붙여 답하세요."}
                    idx = ss.order[ss.pos] + 1
                    ss.questions.insert(idx, new_q)
                    ss.order_sig = None
                    sync_order()
                    if idx in ss.order:
                        ss.pos = ss.order.index(idx)
                    reset_timer()
                    st.rerun()


with tab_practice:
    practice_tab()

# ---------------------------------------------------------------------------
# 2) 맞춤 질문 만들기
# ---------------------------------------------------------------------------
with tab_custom:
    st.markdown("### 지원 정보")
    c1, c2, c3 = st.columns(3)
    ss.company = c1.text_input("지원 회사", value=ss.company, placeholder="예: ○○식품")
    ss.role = c2.text_input("지원 직무", value=ss.role)
    region = c3.text_input("담당 지역·시장", value="동남아 (베트남·인도네시아·태국 등)")
    jd = st.text_area("채용공고나 회사 정보 (붙여 넣기)", height=140,
                      placeholder="주요 제품, 수출 국가, 우대 사항, 인재상 등을 붙여 넣으면 질문이 더 정확해져요.")
    profile = st.text_area("내 경험과 강점", height=120,
                           placeholder="예: 무역학과, 무역 실무 교육 수료 프로젝트, 어학 점수, 아르바이트 경험 등")

    st.markdown("### 1단계 · 실제 면접 질문 모으기")
    st.caption("공개된 면접 후기를 웹에서 찾아 출처와 함께 보여 줘요. 후기 사이트를 긁어 오지 않고 AI 웹 검색 결과만 사용하니, 링크를 열어 직접 확인하세요.")
    if st.button("실제 면접 질문 찾기", disabled=not ss.company.strip()) and ai_allowed():
        with st.spinner("웹에서 면접 후기를 찾는 중... (30초~1분)"):
            try:
                guard.count_ai_use()
                found = llm.collect_real_questions(ss.company.strip(), ss.role.strip())
                ss.collected, ss.citations = found["questions"], found["citations"]
                if not ss.collected:
                    st.warning("공개된 면접 질문을 찾지 못했어요. 비슷한 회사 이름이나 직무로 다시 찾거나, 아래에 직접 붙여 넣으세요.")
            except llm.AIError as e:
                st.error(str(e))
    if not ss.company.strip():
        st.caption("지원 회사를 먼저 입력하세요.")

    picked_collected = []
    if ss.collected:
        df = pd.DataFrame([{"연습에 넣기": True, "분류": c.get("category", "직무"), "질문": c["question"],
                            "출처": c.get("source_url", "")} for c in ss.collected])
        edited = st.data_editor(df, hide_index=True, use_container_width=True,
                                column_config={"출처": st.column_config.LinkColumn("출처")},
                                disabled=["분류", "질문", "출처"], key="collected_editor")
        picked_collected = [ss.collected[i] for i, row in edited.iterrows() if row["연습에 넣기"]]
        if ss.citations:
            with st.expander("AI 가 참고한 글"):
                for c in ss.citations:
                    st.markdown(f"- [{c['title']}]({c['url']})")

    manual = st.text_area("직접 모은 질문 (한 줄에 하나)", height=100,
                          placeholder="면접 후기나 선배에게 들은 질문을 한 줄씩 붙여 넣으세요.")
    manual_qs = [line.strip(" -•\t") for line in manual.splitlines() if line.strip()]

    st.markdown("### 2단계 · AI 예상 질문 만들기")
    n1, n2, n3 = st.columns(3)
    counts = {"인성": n1.number_input("인성", 0, 15, 8), "직무": n2.number_input("직무", 0, 15, 8),
              "영어": n3.number_input("영어", 0, 10, 4)}
    if st.button("AI 예상 질문 만들기", type="primary") and ai_allowed():
        with st.spinner("회사와 직무에 맞는 질문을 만드는 중..."):
            try:
                guard.count_ai_use()
                gen = llm.generate_questions(company=ss.company, role=ss.role, region=region, jd_text=jd,
                                             profile=profile, counts=counts,
                                             collected=[c["question"] for c in picked_collected] + manual_qs)
                real = [{"category": c.get("category", "직무") if c.get("category") in ("인성", "직무", "영어") else "직무",
                         "question": c["question"], "tip": "실제 면접 후기에 나온 질문이에요."} for c in picked_collected]
                real += [{"category": "직무", "question": m, "tip": "직접 모은 질문이에요."} for m in manual_qs]
                ss.generated = real + gen
                st.success(f"질문 {len(ss.generated)}개를 만들었어요. 아래에서 확인한 뒤 연습을 시작하세요.")
            except llm.AIError as e:
                st.error(str(e))

    if ss.generated:
        st.dataframe(pd.DataFrame([{"분류": g["category"], "질문": g["question"],
                                    "확인하려는 것": g.get("reason", "") or g.get("tip", "")} for g in ss.generated]),
                     hide_index=True, use_container_width=True)
        st.button("이 질문으로 연습 시작", type="primary", on_click=use_question_set,
                  args=([dict(g) for g in ss.generated], "맞춤 질문"))
        if ss.set_name == "맞춤 질문":
            st.success("'연습하기' 탭에서 맞춤 질문으로 연습하고 있어요.")

# ---------------------------------------------------------------------------
# 3) 종합 리포트
# ---------------------------------------------------------------------------
with tab_report:
    if not ss.results:
        st.info("아직 분석한 답변이 없어요. '연습하기'에서 녹음 후 'AI 분석 받기'를 눌러 주세요.")
    else:
        items = list(ss.results.values())
        avg = lambda key: round(sum(key(r) for r in items) / len(items))  # noqa: E731
        m = st.columns(4)
        m[0].metric("답변한 질문", f"{len(items)}개")
        m[1].metric("평균 종합", f'{avg(lambda r: r["total"])}점')
        m[1].caption(grade(avg(lambda r: r["total"])))
        m[2].metric("평균 내용", avg(lambda r: r["ai"].get("overall", 0)))
        m[3].metric("평균 자신감", avg(lambda r: r["audio"]["confidence_score"]))

        rows = summary_rows(ss.results)
        chart = pd.DataFrame({"종합": [r["종합"] for r in rows]}, index=[f"Q{i}" for i in range(1, len(rows) + 1)])
        st.markdown("#### 질문별 종합 점수")
        st.bar_chart(chart, color=NAVY)

        crit = {}
        for r in items:
            for k, v in r["ai"].get("scores", {}).items():
                crit.setdefault(k, []).append(v)
        if crit:
            st.markdown("#### 평가 항목별 평균")
            st.bar_chart(pd.DataFrame({"평균": {k: round(sum(v) / len(v)) for k, v in crit.items()}}),
                         horizontal=True, color=NAVY)

        fillers = {}
        for r in items:
            for k, v in {**r["audio"]["fillers"], **r["audio"]["habits"]}.items():
                fillers[k] = fillers.get(k, 0) + v
        if fillers:
            st.markdown("#### 자주 나온 습관어")
            st.write(", ".join(f"'{k}' {v}회" for k, v in sorted(fillers.items(), key=lambda x: -x[1])[:8]))

        st.markdown("#### 질문별 상세")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if st.button("AI 종합 총평 만들기") and ai_allowed():
            with st.spinner("전체 답변을 돌아보는 중..."):
                try:
                    brief = [{"질문": r["question"], "종합": r["total"], "총평": r["ai"].get("summary"),
                              "고칠 점": r["ai"].get("improvements"), "습관어": r["audio"]["filler_total"],
                              "말하기 속도": r["audio"]["speech_rate_label"], "톤": r["audio"]["tone_label"]}
                             for r in items]
                    guard.count_ai_use()
                    ss.ai_summary = llm.overall_summary(role=ss.role, company=ss.company, rows=brief)
                except llm.AIError as e:
                    st.error(str(e))
        if ss.ai_summary:
            st.markdown(ss.ai_summary)

        md = build_markdown(ss.results, role=ss.role, company=ss.company, ai_summary=ss.ai_summary)
        d1, d2 = st.columns(2)
        d1.download_button("리포트 내려받기 (.md)", md, file_name="면접_연습_리포트.md",
                           mime="text/markdown", use_container_width=True, type="primary")
        if d2.button("기록 모두 지우기", use_container_width=True):
            ss.results, ss.ai_summary = {}, ""
            st.rerun()
