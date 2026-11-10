"""OpenAI 연동: 음성 받아쓰기, 답변 평가, 질문 수집·생성, 종합 총평."""
from __future__ import annotations

import io
import json
import re

from core import settings

CHAT_MODEL = settings.get("OPENAI_MODEL", "gpt-4o-mini")
TRANSCRIBE_MODEL = settings.get("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
SEARCH_MODEL = settings.get("OPENAI_SEARCH_MODEL", CHAT_MODEL)

# 받아쓰기 모델은 "음, 어" 같은 소리를 지우는 경향이 있어, 그대로 적도록 예시를 줍니다.
VERBATIM_PROMPT = {
    "ko": "음, 어... 그, 저는 음 해외영업에 지원하게 된, 어, 이유가 있는데요. 그니까 약간, 이제...",
    "en": "Um, uh, so... a, I mean, like, I was, uh, working on a... you know, a project.",
}


class AIError(RuntimeError):
    """화면에 그대로 보여 줄 수 있는 한국어 오류."""


def _api_key() -> str:
    return settings.get("OPENAI_API_KEY")


def has_api_key() -> bool:
    return not settings.is_placeholder(_api_key())


def get_client():
    if not has_api_key():
        raise AIError("OpenAI API 키가 없어요. 내 PC에서는 .env 파일에, Streamlit Cloud 에서는 Secrets 에 "
                      "OPENAI_API_KEY 를 넣고 앱을 다시 실행하세요.")
    from openai import OpenAI
    return OpenAI(api_key=_api_key())


def _parse_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            return json.loads(m.group(1))
        raise AIError("AI 응답을 읽지 못했어요. 한 번 더 시도해 주세요.")


def _chat_json(system: str, user: str, temperature: float = 0.4) -> dict:
    client = get_client()
    try:
        res = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    except Exception as e:  # noqa: BLE001
        raise AIError(f"OpenAI 요청이 실패했어요: {e}") from e
    return _parse_json(res.choices[0].message.content or "{}")


# ---------------------------------------------------------------------------
# 1) 음성 → 글자
# ---------------------------------------------------------------------------
def transcribe(wav_bytes: bytes, lang: str = "ko") -> str:
    client = get_client()
    f = io.BytesIO(wav_bytes)
    f.name = "answer.wav"
    try:
        res = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL, file=f, language=lang,
            prompt=VERBATIM_PROMPT.get(lang, ""), temperature=0,
        )
    except Exception as e:  # noqa: BLE001
        raise AIError(f"음성 받아쓰기에 실패했어요: {e}") from e
    return (getattr(res, "text", "") or "").strip()


# ---------------------------------------------------------------------------
# 2) 답변 평가
# ---------------------------------------------------------------------------
EVAL_SYSTEM = """당신은 한국 무역·제조 기업의 해외영업 신입 채용 면접관입니다.
지원자의 실제 답변(음성을 받아쓴 글)과 전달 지표를 보고, 엄격하지만 도움이 되는 피드백을 줍니다.
규칙:
- 받아쓴 글에 실제로 있는 내용만 근거로 평가합니다. 없는 경험을 지어내지 않습니다.
- 답변이 매우 짧거나 질문과 관계없으면 점수를 낮게 주고 그 이유를 분명히 적습니다.
- 점수는 0~100 정수. 70점은 '무난한 신입 답변', 85점 이상은 '바로 합격권'입니다.
- 모든 설명은 한국어로 씁니다. 단, 영어 질문의 '개선 답변 예시'는 영어로 씁니다.
- 반드시 아래 JSON 형식만 출력합니다."""

EVAL_FORMAT = """{
  "scores": {"논리성": 0, "구체성": 0, "직무 적합성": 0, "답변 구조": 0, "질문 이해도": 0},
  "overall": 0,
  "summary": "한 문장 총평",
  "logic_flow": "답변이 어떤 순서로 전개됐는지와 논리가 끊긴 지점",
  "strengths": ["잘한 점"],
  "improvements": ["고칠 점 (구체적인 방법 포함)"],
  "better_answer": "같은 경험을 살린 개선 답변 예시 (60초 분량)",
  "follow_up": "면접관이 이어서 물어볼 꼬리 질문 1개",
  "english_feedback": ["영어 질문일 때만: 어색한 표현 → 자연스러운 표현"]
}"""


def evaluate_answer(*, question: str, category: str, tip: str, transcript: str,
                    role: str, company: str = "", delivery: dict | None = None) -> dict:
    user = f"""[지원 회사] {company or '미정'}
[지원 직무] {role}
[질문 분류] {category}
[질문] {question}
[면접관이 기대하는 답변 포인트] {tip}

[지원자 답변 받아쓰기]
{transcript or '(말소리가 거의 없음)'}

[전달 지표 - 참고용]
{json.dumps(delivery or {}, ensure_ascii=False)}

아래 형식으로 평가하세요.
{EVAL_FORMAT}"""
    data = _chat_json(EVAL_SYSTEM, user, temperature=0.3)
    scores = data.get("scores", {}) or {}
    data["scores"] = {k: int(max(0, min(100, float(v)))) for k, v in scores.items() if _is_num(v)}
    if not _is_num(data.get("overall")) and data["scores"]:
        data["overall"] = round(sum(data["scores"].values()) / len(data["scores"]))
    data["overall"] = int(float(data.get("overall", 0)))
    if category != "영어":
        data["english_feedback"] = []
    return data


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 3) 실제 면접 질문 수집 (웹 검색)
# ---------------------------------------------------------------------------
def collect_real_questions(company: str, role: str) -> dict:
    """공개된 면접 후기에서 실제 질문을 찾습니다. 출처 링크와 함께 돌려줍니다."""
    client = get_client()
    prompt = f"""'{company}' 회사의 '{role}' (또는 비슷한 해외영업·무역 직무) 면접에서
실제로 나왔다고 공개된 질문을 웹에서 찾아 주세요.
잡코리아·사람인·잡플래닛·캐치·링커리어 면접 후기, 회사 채용 블로그·뉴스 인터뷰처럼 공개된 글만 사용합니다.
규칙:
- 실제 글에서 확인한 질문만 적고, 추측해서 만들지 않습니다. 찾지 못하면 빈 목록을 돌려줍니다.
- 같은 뜻의 질문은 하나로 합칩니다. 최대 15개.
- JSON 만 출력합니다: {{"questions": [{{"question": "질문", "category": "인성|직무|영어", "source_title": "글 제목", "source_url": "https://..."}}]}}"""
    last_err = None
    for tool_type in ("web_search", "web_search_preview"):
        try:
            res = client.responses.create(model=SEARCH_MODEL, tools=[{"type": tool_type}], input=prompt)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        raise AIError(f"웹 검색 기능을 쓸 수 없어요 ({last_err}). .env 의 OPENAI_SEARCH_MODEL 을 웹 검색을 지원하는 모델로 바꾸거나, 아래 칸에 직접 모은 질문을 붙여 넣으세요.")

    text = getattr(res, "output_text", "") or ""
    try:
        data = _parse_json(text)
    except AIError:
        data = {"questions": []}
    questions = data.get("questions", []) if isinstance(data, dict) else data

    # 응답에 붙은 실제 인용 링크도 함께 모읍니다.
    cited = []
    for item in getattr(res, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                url = getattr(ann, "url", None)
                if url and url not in [c["url"] for c in cited]:
                    cited.append({"title": getattr(ann, "title", url), "url": url})
    clean = [q for q in questions if isinstance(q, dict) and q.get("question")]
    return {"questions": clean, "citations": cited}


# ---------------------------------------------------------------------------
# 4) 맞춤 예상 질문 생성
# ---------------------------------------------------------------------------
GEN_SYSTEM = """당신은 한국 무역·제조 기업 해외영업 직무의 면접 코치입니다.
지원 정보에 꼭 맞는 예상 면접 질문을 만듭니다. 질문과 설명은 한국어로 씁니다.
영어 질문은 '영어로, ~해 보세요' 처럼 한국어로 지시문을 쓰고, 답변은 영어로 하도록 합니다.
반드시 JSON 만 출력합니다."""


def generate_questions(*, company: str, role: str, region: str, jd_text: str, profile: str,
                       counts: dict, collected: list[str] | None = None) -> list[dict]:
    collected_txt = "\n".join(f"- {q}" for q in (collected or [])) or "(없음)"
    user = f"""[지원 회사] {company}
[지원 직무] {role}
[담당 지역·시장] {region}
[채용공고·회사 정보]
{jd_text or '(없음)'}
[지원자 정보 - 경험, 강점, 자격]
{profile or '(없음)'}
[실제로 나왔던 질문 - 참고만 하고 그대로 베끼지 말 것]
{collected_txt}

인성 {counts.get('인성', 0)}개, 직무 {counts.get('직무', 0)}개, 영어 {counts.get('영어', 0)}개를 만드세요.
- 회사의 제품·수출 국가·사업 방향, 지원자 경험을 질문에 직접 넣어 '이 회사에서만 나올 질문'으로 만듭니다.
- 각 질문마다 면접관이 확인하려는 것(reason)과 답변 포인트(tip)를 한 문장씩 씁니다.
형식: {{"questions": [{{"category": "인성|직무|영어", "question": "...", "tip": "...", "reason": "..."}}]}}"""
    data = _chat_json(GEN_SYSTEM, user, temperature=0.7)
    out = []
    for q in data.get("questions", []):
        if isinstance(q, dict) and q.get("question"):
            cat = q.get("category", "직무")
            out.append({"category": cat if cat in ("인성", "직무", "영어") else "직무",
                        "question": q["question"].strip(),
                        "tip": (q.get("tip") or "").strip(),
                        "reason": (q.get("reason") or "").strip()})
    if not out:
        raise AIError("질문을 만들지 못했어요. 입력 내용을 조금 더 채운 뒤 다시 시도해 주세요.")
    return out


# ---------------------------------------------------------------------------
# 5) 종합 총평
# ---------------------------------------------------------------------------
def overall_summary(*, role: str, company: str, rows: list[dict]) -> str:
    client = get_client()
    user = f"""[지원] {company or '미정'} / {role}
[질문별 결과]
{json.dumps(rows, ensure_ascii=False)}

위 결과로 지원자의 면접 준비 상태를 한국어로 정리하세요. 마크다운으로,
1) 전체 평가 한 문단  2) 반복해서 보인 강점 3가지  3) 반복해서 보인 약점 3가지(근거 질문 포함)
4) 다음 3일 연습 계획 (하루 단위, 구체적으로). 과장하지 말고 결과에 있는 내용만 쓰세요."""
    try:
        res = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.4,
            messages=[{"role": "system", "content": "당신은 해외영업 직무 면접 코치입니다."},
                      {"role": "user", "content": user}],
        )
    except Exception as e:  # noqa: BLE001
        raise AIError(f"총평 생성에 실패했어요: {e}") from e
    return res.choices[0].message.content or ""
