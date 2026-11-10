"""점수 합산과 리포트(마크다운) 만들기."""
from __future__ import annotations

from datetime import datetime


def total_score(ai: dict, audio: dict, face: dict | None) -> int:
    content = ai.get("overall", 0)
    voice = audio.get("confidence_score", 0)
    if face and face.get("face_ratio", 0) > 0:
        return round(0.60 * content + 0.25 * voice + 0.15 * face.get("face_score", 0))
    return round(0.70 * content + 0.30 * voice)


def grade(score: float) -> str:
    if score >= 85:
        return "합격권"
    if score >= 70:
        return "무난함"
    if score >= 55:
        return "보완 필요"
    return "집중 연습 필요"


def summary_rows(results: dict) -> list[dict]:
    rows = []
    for r in results.values():
        ai, au, fa = r["ai"], r["audio"], r.get("face") or {}
        row = {
            "분류": r["category"],
            "질문": r["question"],
            "종합": r["total"],
            "내용(AI)": ai.get("overall", 0),
            "자신감(음성)": au.get("confidence_score", 0),
            "표정·시선": fa.get("face_score") if fa.get("face_ratio") else None,
            "말하기 속도": f'{au.get("speech_rate", 0):.0f} {au.get("speech_rate_unit", "")}',
            "습관어(회)": au.get("filler_total", 0),
        }
        row.update({k: v for k, v in ai.get("scores", {}).items()})
        rows.append(row)
    return rows


def build_markdown(results: dict, *, role: str, company: str, ai_summary: str = "") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    items = list(results.values())
    avg = round(sum(r["total"] for r in items) / len(items)) if items else 0
    lines = [
        "# 면접 연습 리포트",
        "",
        f"- 지원: {company or '미정'} / {role}",
        f"- 작성: {now}",
        f"- 답변한 질문: {len(items)}개",
        f"- 평균 종합 점수: **{avg}점 ({grade(avg)})**",
        "",
    ]
    if ai_summary:
        lines += ["## AI 종합 총평", "", ai_summary.strip(), ""]

    lines += ["## 질문별 점수", "", "| 분류 | 질문 | 종합 | 내용 | 자신감 | 표정·시선 | 습관어 |",
              "|---|---|---|---|---|---|---|"]
    for r in items:
        fa = r.get("face") or {}
        face_txt = str(fa.get("face_score")) if fa.get("face_ratio") else "-"
        q = r["question"].replace("|", "/")
        lines.append(f'| {r["category"]} | {q} | {r["total"]} | {r["ai"].get("overall", 0)} | '
                     f'{r["audio"].get("confidence_score", 0)} | {face_txt} | {r["audio"].get("filler_total", 0)}회 |')
    lines.append("")

    for i, r in enumerate(items, 1):
        ai, au, fa = r["ai"], r["audio"], r.get("face") or {}
        lines += [f"## {i}. [{r['category']}] {r['question']}", "",
                  f"**종합 {r['total']}점 · {grade(r['total'])}** — {ai.get('summary', '')}", ""]
        if ai.get("scores"):
            lines.append(" · ".join(f"{k} {v}" for k, v in ai["scores"].items()))
            lines.append("")
        lines += ["### 내 답변 (받아쓰기)", "", f"> {r['transcript'] or '(말소리 없음)'}", ""]
        lines += ["### 논리 흐름", "", ai.get("logic_flow", "-"), ""]
        lines += ["### 잘한 점"] + [f"- {s}" for s in ai.get("strengths", [])] + [""]
        lines += ["### 고칠 점"] + [f"- {s}" for s in ai.get("improvements", [])] + [""]
        if ai.get("english_feedback"):
            lines += ["### 영어 표현 교정"] + [f"- {s}" for s in ai["english_feedback"]] + [""]
        lines += ["### 전달력",
                  f"- 말하기 속도: {au.get('speech_rate', 0):.0f} {au.get('speech_rate_unit', '')} ({au.get('speech_rate_label', '-')})",
                  f"- 습관어: {au.get('filler_total', 0)}회 (분당 {au.get('filler_per_min', 0)}회) "
                  + _fmt_counts({**au.get('fillers', {}), **au.get('habits', {})}),
                  f"- 목소리: {au.get('tone_label', '-')}, 크기 {au.get('volume_label', '-')}, "
                  f"멈춤 {au.get('pause_count', 0)}회 (최장 {au.get('longest_pause_sec', 0)}초)",
                  f"- 자신감 점수: {au.get('confidence_score', 0)}점"]
        lines += [f"  - {n}" for n in au.get("confidence_notes", [])]
        if fa.get("face_ratio"):
            lines += [f"- 표정·시선 점수: {fa.get('face_score', 0)}점 "
                      f"(카메라 응시 {round(fa.get('eye_contact_ratio', 0) * 100)}%, "
                      f"미소 {round(fa.get('smile_ratio', 0) * 100)}%, 깜빡임 분당 {fa.get('blinks_per_min', 0)}회)"]
            lines += [f"  - {n}" for n in fa.get("notes", []) or []]
        lines += ["", "### 개선 답변 예시", "", ai.get("better_answer", "-"), "",
                  f"**예상 꼬리 질문:** {ai.get('follow_up', '-')}", ""]
    return "\n".join(lines)


def _fmt_counts(d: dict) -> str:
    if not d:
        return ""
    return "(" + ", ".join(f"'{k}' {v}회" for k, v in sorted(d.items(), key=lambda x: -x[1])) + ")"
