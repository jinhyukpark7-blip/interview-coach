# AI 면접 연습기 (해외영업 신입 · 동남아)

카드로 예상 질문을 넘기며 **카메라와 마이크로 실제 면접처럼 연습**하고,
AI 가 답변 내용·말하기 속도·습관어·목소리 자신감·표정과 시선을 점수로 알려 주는 Streamlit 앱입니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 질문 카드 | 기본 20개(인성 8 · 직무 8 · 영어 4), 이전·다음·섞기, 분류별 보기, 답변 포인트 |
| 60초 타이머 | 답변 시작·멈춤·처음으로, 10초 남으면 주황색 |
| 답변 메모 | 질문마다 따로 적는 메모. **화면에만 있고 저장되지 않음** |
| 카메라 실시간 연습 | MediaPipe 얼굴 인식. 카메라 응시·정면 자세·미소·눈 깜빡임·고개 흔들림 측정 |
| 마이크 답변 | 녹음 → OpenAI 로 받아쓰기 → 분석 |
| 말하기 분석 | 말하기 속도(한국어 음절/분, 영어 단어/분), 멈춤 횟수·가장 긴 멈춤 |
| 습관어 체크 | 한국어 `음, 어, 아, 그…, 저…, 약간, 이제, 막` / 영어 `um, uh, er, ah, a…, like, you know` 등 |
| 목소리 톤·자신감 | 목소리 크기, 억양 변화, 멈춤 비율, 습관어, 속도를 합친 자신감 점수 |
| GPT 피드백 | 논리성·구체성·직무 적합성·답변 구조·질문 이해도 점수, 잘한 점·고칠 점, 개선 답변 예시, 꼬리 질문 |
| 맞춤 질문 | 회사·직무·채용공고·내 경험 입력 → 공개된 실제 면접 후기 찾기(출처 링크) → AI 예상 질문 생성 |
| 종합 리포트 | 질문별 점수 차트, 항목별 평균, 자주 나온 습관어, AI 종합 총평, 리포트 내려받기(.md) |
| **공개용 안전장치** | 비밀번호 로그인, AI 사용 한도(하루·접속당), 녹음 길이 제한, 개인정보 안내 |

---

# 1부. 내 PC에서 실행하기 (Windows)

## 1. 준비물
- Python 3.10 이상 (3.12 권장. 3.14 에서도 설치되는 것을 확인함)
- VS Code, 웹캠, 마이크

## 2. 가상환경 만들고 패키지 설치

VS Code 에서 `보기 → 터미널` 을 열고 한 줄씩 입력합니다. (`cmd` 터미널 기준)

```
python -m venv .venv
.venv\Scripts\activate
cd interview_coach
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- 줄 앞에 `(.venv)` 가 붙어 있어야 해요.
- `.venv` 를 `interview_coach` 바깥에 만든 경우, 위처럼 `cd interview_coach` 로 들어간 뒤 설치·실행합니다.

## 3. API 키 넣기 (.env)

1. https://platform.openai.com/api-keys → **Create new secret key** → `sk-...` 키 복사
   - API 는 쓴 만큼 요금이 나가요. **Billing** 에 결제 수단이나 충전 금액이 있어야 동작합니다 (ChatGPT 구독과 별개).
2. VS Code 에서 `interview_coach/.env` 를 열고 첫 줄을 바꿉니다.
   ```
   OPENAI_API_KEY=sk-복사한키전체
   ```
3. `Ctrl + S` 저장.

> `.env` 는 비밀번호와 같아요. 캡처해서 올리거나 GitHub 에 올리지 마세요. (`.gitignore` 에 들어 있음)
> 키가 없어도 앱은 열려요. 카드·타이머·메모·카메라는 되고, AI 분석 버튼만 안내 문구가 뜹니다.

## 4. 실행

```
streamlit run app.py
```
- `Email:` 을 물어보면 그냥 Enter.
- 브라우저가 열리지 않으면 http://localhost:8501 접속.
- 끌 때는 터미널에서 `Ctrl + C`.

**다음에 다시 켤 때**
```
.venv\Scripts\activate
cd interview_coach
streamlit run app.py
```

---

# 2부. 인터넷에 공개하기 (GitHub → Streamlit Community Cloud)

전체 흐름: **① OpenAI 사용 한도 설정 → ② GitHub 에 코드 올리기 → ③ Streamlit 에서 앱 만들기 + Secrets 에 키 넣기 → ④ 확인**

## ① OpenAI 쪽 안전장치 먼저

앱 안에도 사용 한도가 있지만, 앱이 다시 시작되면 숫자가 0으로 돌아가요. **진짜 마지막 방어선은 OpenAI 계정의 예산 한도**입니다.

1. https://platform.openai.com 로그인 → 오른쪽 위 **설정(톱니바퀴)**
2. **Limits**(또는 Billing → Usage limits) 메뉴에서 **월 예산(Budget)** 을 작게 설정 (예: 5달러)
3. 배포용 키는 **새로 하나 따로 만들기**를 권장해요. 문제가 생기면 그 키만 지우면 됩니다.

## ② GitHub 에 올리기 (웹 브라우저만 사용)

### 2-1. 계정과 저장소 만들기
1. https://github.com 가입·로그인
2. 오른쪽 위 **+** → **New repository**
3. 입력
   - Repository name: `interview-coach` (영어·숫자·하이픈만)
   - **Public** 선택 (키는 코드에 없으니 공개해도 괜찮아요)
   - 나머지는 그대로 두고 **Create repository**

### 2-2. 파일 올리기
1. 새 저장소 화면에서 **uploading an existing file** 링크를 누릅니다.
2. 파일 탐색기에서 `C:\Users\user\AX2\20261110_VibeCoding\interview_coach` 폴더를 엽니다.
3. **아래 항목만** 선택해서 브라우저 창으로 끌어다 놓습니다.

   | 올리기 ✅ | 올리지 않기 ❌ |
   |---|---|
   | `app.py` | **`.env`** (키가 들어 있음) |
   | `core` 폴더 | `.venv` 폴더 |
   | `.streamlit` 폴더 | `models` 폴더 |
   | `requirements.txt` | `__pycache__` 폴더 |
   | `packages.txt` | |
   | `README.md`, `.gitignore`, `.env.example` | |

   > 웹으로 올릴 때는 `.gitignore` 가 적용되지 않아요. **`.env` 를 빼는 건 직접 해야** 합니다.
4. 아래쪽 **Commit changes** 클릭.
5. 저장소 첫 화면에 `app.py`, `core`, `.streamlit`, `requirements.txt`, `packages.txt` 가 **맨 바깥에** 보이면 성공.
   - `.env` 가 보이면 즉시 그 파일을 열고 휴지통 아이콘으로 지운 뒤, OpenAI 에서 그 키를 **삭제하고 새로 발급**하세요.

### 2-3. 나중에 코드를 고쳤을 때
같은 저장소에서 **Add file → Upload files** 로 바뀐 파일을 다시 끌어다 놓고 **Commit changes**. Streamlit 앱이 몇 분 안에 자동으로 새 코드로 바뀝니다.

## ③ Streamlit Community Cloud 에 배포

1. https://share.streamlit.io 접속 → **Continue with GitHub** 로 로그인 → GitHub 연결 허용
2. 오른쪽 위 **Create app** → **Deploy a public app from GitHub**(또는 "Yup, I have an app")
3. 입력
   - Repository: `내아이디/interview-coach`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: 원하는 주소 (예: `jh-interview-coach`)
4. **Advanced settings** 를 엽니다.
   - **Python version**: `3.12` 선택
   - **Secrets** 칸에 아래를 붙여 넣고 값만 바꿉니다. (`.streamlit/secrets.toml.example` 과 같은 내용)
     ```toml
     OPENAI_API_KEY = "sk-내키"
     APP_PASSWORD = "내가정한비밀번호"
     DAILY_AI_LIMIT = "60"
     SESSION_AI_LIMIT = "30"
     MAX_RECORD_SEC = "180"
     ```
     - 값은 **큰따옴표로 감싸야** 해요.
     - `APP_PASSWORD` 는 꼭 넣으세요. 이걸 아는 사람만 앱에 들어올 수 있어요.
   - **Save**
5. **Deploy** 클릭 → 처음에는 패키지 설치로 5~10분 걸려요. 오른쪽 아래 **Manage app** 에서 진행 기록을 볼 수 있어요.
6. 비밀번호 화면이 뜨면 성공. 주소는 `https://jh-interview-coach.streamlit.app` 같은 형태예요.

**Secrets 를 나중에 바꾸려면:** share.streamlit.io → 내 앱 오른쪽 **⋮** → **Settings → Secrets** → 수정 → **Save** (앱이 자동으로 다시 시작됨)

## ④ 공개 후 확인 목록

- [ ] 비밀번호 없이는 들어갈 수 없다
- [ ] 사이드바에 "OpenAI 연결됨" 이 보인다
- [ ] 녹음 → AI 분석 받기가 된다
- [ ] 사이드바의 AI 사용 횟수가 늘어난다
- [ ] 카메라를 켜면 내 얼굴과 테두리가 보인다 (안 되면 아래 '카메라가 연결되지 않을 때')
- [ ] GitHub 저장소에 `.env` 가 없다

### 카메라가 연결되지 않을 때 (클라우드에서 자주 생김)
회사·학교 와이파이나 일부 통신망에서는 카메라 영상이 Streamlit 서버까지 가지 못해 화면이 검은색으로 남아요. 이럴 때는 **TURN 서버**가 필요합니다.

1. 무료 TURN 서비스(예: Metered, Twilio)에 가입해 TURN 주소·아이디·비밀번호를 받습니다.
2. Streamlit Secrets 에 추가합니다.
   ```toml
   TURN_URL = "turn:받은주소:80,turn:받은주소:443"
   TURN_USERNAME = "받은아이디"
   TURN_CREDENTIAL = "받은비밀번호"
   ```
3. Save 후 다시 시도. 카메라 없이도 녹음·AI 분석은 정상 동작합니다.

---

## 안전장치 설정값

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `APP_PASSWORD` | (없음) | 앱 비밀번호. 비워 두면 비밀번호 없이 열림. 공개할 때는 필수 |
| `DAILY_AI_LIMIT` | 60 | 앱 전체에서 하루에 쓸 수 있는 AI 요청 수 |
| `SESSION_AI_LIMIT` | 30 | 한 번 접속해서 쓸 수 있는 AI 요청 수 |
| `MAX_RECORD_SEC` | 180 | 녹음 최대 길이(초). 길면 분석하지 않음 |

- AI 요청 1회: 답변 분석 1번, 실제 질문 찾기 1번, 예상 질문 만들기 1번, 종합 총평 1번 각각.
- 비밀번호를 5번 틀리면 5분 동안 잠깁니다.
- 앱 안의 한도 숫자는 앱이 다시 시작되면 0으로 돌아가요. 그래서 OpenAI 예산 한도를 함께 설정해야 합니다.

## 점수 계산 방식

- **종합 점수** = 내용(AI) 60% + 목소리 자신감 25% + 표정·시선 15% (카메라를 끄면 내용 70% + 자신감 30%)
- **말하기 속도 적정 범위**: 한국어 250~330 음절/분, 영어 120~160 단어/분
- 기준값은 `core/audio_analysis.py` 위쪽 `RATE_RANGE`, 습관어 목록은 `KO_FILLERS`, `EN_FILLERS` 에서 바꿀 수 있어요.

## 개인정보와 한계

- **카메라 영상**: 앱이 실행되는 곳(내 PC 또는 Streamlit 서버)에서 실시간으로 얼굴 위치·시선만 계산하고 저장하지 않아요. OpenAI 로 보내지 않아요.
- **녹음**: 받아쓰기와 평가를 위해 OpenAI 로 전송돼요. 앱은 음성 파일을 따로 저장하지 않아요.
- **메모·결과**: 열어 둔 화면에만 있고, 새로고침하면 사라져요.
- **습관어 정확도**: 받아쓰기 모델이 "음, 어" 를 지우는 경우가 있어 실제보다 적게 셀 수 있어요. 영어 `a` 는 관사와 구분하려고 `a…`, `a,` 처럼 끊길 때만 셉니다.
- **자신감·톤 점수**는 목소리 크기·억양·멈춤으로 계산한 연습용 참고 지표예요.
- **실제 면접 질문 찾기**는 후기 사이트를 직접 긁어 오지 않고 OpenAI 웹 검색 결과만 씁니다. 출처 링크를 열어 꼭 확인하세요.

## 문제 해결

| 증상 | 해결 |
|---|---|
| `OpenAI 키가 없어요` | 내 PC: `.env` 확인 후 앱 재실행 / 클라우드: Secrets 확인 (큰따옴표) |
| `requirements.txt` 를 찾을 수 없음 | 터미널이 `interview_coach` 폴더 안인지 확인 (`cd interview_coach`) |
| 클라우드 배포 중 `libGL.so.1` 오류 | 저장소 맨 바깥에 `packages.txt` 가 있는지 확인 |
| 클라우드 배포 중 `mediapipe` 설치 실패 | Advanced settings 에서 Python 3.12 선택 후 다시 배포 (앱 삭제 후 재생성) |
| 카메라 화면이 검은색 | 브라우저 카메라 권한 허용, 다른 앱이 카메라를 쓰는지 확인, 클라우드면 위 TURN 설정 |
| 얼굴 인식 모델 다운로드 실패 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task 를 받아 `models/face_landmarker.task` 로 저장 |
| 녹음 버튼이 없음 | `pip install -U streamlit` |
| 클라우드 앱이 잠들어 있음 | 무료 플랜은 한동안 안 쓰면 잠들어요. 화면의 깨우기 버튼을 누르면 1~2분 뒤 다시 켜짐 |

## 폴더 구조

```
interview_coach/
├─ app.py                        # Streamlit 화면
├─ core/
│  ├─ questions.py               # 기본 질문 20개
│  ├─ llm.py                     # OpenAI: 받아쓰기, 평가, 질문 수집·생성, 총평
│  ├─ audio_analysis.py          # 말하기 속도, 습관어, 톤, 자신감
│  ├─ face_analysis.py           # MediaPipe 얼굴·시선 분석
│  ├─ report.py                  # 점수 합산, 리포트
│  ├─ guard.py                   # 비밀번호, 사용 한도
│  └─ settings.py                # .env / Secrets 읽기
├─ .streamlit/
│  ├─ config.toml                # 남색·흰색 테마
│  └─ secrets.toml.example       # 클라우드 Secrets 예시
├─ packages.txt                  # 클라우드 서버용 시스템 패키지
├─ requirements.txt
├─ .env                          # 내 PC 용 API 키 (GitHub 에 올리지 않음)
├─ .env.example
└─ README.md
```
