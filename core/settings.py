"""설정값 읽기: 내 PC에서는 .env, Streamlit Cloud 에서는 Secrets 에서 읽습니다."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        try:
            import streamlit as st
            if name in st.secrets:
                value = str(st.secrets[name])
        except Exception:  # noqa: BLE001  (secrets.toml 이 없으면 그냥 넘어감)
            value = None
    value = (value or "").strip()
    return value if value else default


def get_int(name: str, default: int) -> int:
    try:
        return int(get(name, str(default)))
    except ValueError:
        return default


def is_placeholder(value: str) -> bool:
    return not value or value.startswith("여기에")
