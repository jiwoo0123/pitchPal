import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from deepface import DeepFace
import cv2
import time
import threading
from collections import Counter
import matplotlib.pyplot as plt
import requests
import streamlit_authenticator as stauth
import uuid
from datetime import datetime
from st_audiorec import st_audiorec

# 사용자 정보 및 streamlit_authenticator 관련 코드 전체 제거

# 로그인 상태 관리
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
if 'login_email' not in st.session_state:
    st.session_state['login_email'] = ''

# 사이드바 로그인 상태 메시지 및 로그인 폼
if st.session_state['login_status']:
    st.sidebar.success(f"👋 {st.session_state['login_email']}님 환영합니다!")
    if st.sidebar.button("로그아웃"):
        st.session_state['login_status'] = False
        st.session_state['login_email'] = ''
        st.rerun()
else:
    st.sidebar.info("로그인해주세요.")
    sidebar_email = st.sidebar.text_input("이메일", key="sidebar_login_email")
    sidebar_pw = st.sidebar.text_input("비밀번호", type="password", key="sidebar_login_pw")
    if st.sidebar.button("로그인", key="sidebar_login_btn"):
        if sidebar_email and sidebar_pw:
            try:
                resp = requests.post("http://localhost:8002/login",
                                     json={"email": sidebar_email, "password": sidebar_pw})
                if resp.status_code == 200:
                    st.session_state['login_status'] = True
                    st.session_state['login_email'] = sidebar_email
                    st.sidebar.success("로그인 성공!")
                    st.rerun()
                else:
                    st.sidebar.error(resp.json().get("detail", "로그인 실패"))
            except Exception as e:
                st.sidebar.error(f"서버 오류: {e}")
        else:
            st.sidebar.warning("이메일과 비밀번호를 모두 입력하세요.")


# 감정 분석용 비디오 트랜스포머
class FaceEmotionAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = "알 수 없음"
        self.emotion_log = []
        self.recording = False
        self.last_log_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            self.last_emotion = emotion

            # 실시간 영상에 감정 텍스트 그리기
            cv2.putText(img, f"Emotion: {emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 감정 기록하기 (5초마다)
            if self.recording and time.time() - self.last_log_time > 5:
                self.emotion_log.append(emotion)
                self.last_log_time = time.time()
        except Exception as e:
            print("감정 분석 실패:", e)

        return img


# 페이지 선택 사이드바
st.sidebar.title("📌 PitchPal 메뉴")
page = st.sidebar.selectbox("이동할 페이지를 선택하세요",
                            ("🏠 메인화면", "🎤 면접 연습", "✍️ 자기소개서 분석", "👤 마이페이지", "🔐 회원가입/비밀번호 찾기"))

# 페이지별 콘텐츠 렌더링
if page == "🏠 메인화면":
    st.title("🏠 PitchPal에 오신 것을 환영합니다!")
    st.write("AI 면접 연습 + 자기소개서 분석 웹앱입니다.")
    if st.session_state['login_status']:
        st.success(f"{st.session_state['login_email']}님 환영합니다!")
    else:
        st.info("로그인 후 다양한 기능을 이용할 수 있습니다.")
        home_email = st.text_input("이메일", key="home_login_email")
        home_pw = st.text_input("비밀번호", type="password", key="home_login_pw")
        if st.button("로그인", key="home_login_btn"):
            if home_email and home_pw:
                try:
                    resp = requests.post("http://localhost:8002/login", json={"email": home_email, "password": home_pw})
                    if resp.status_code == 200:
                        st.session_state['login_status'] = True
                        st.session_state['login_email'] = home_email
                        st.success("로그인 성공!")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "로그인 실패"))
                except Exception as e:
                    st.error(f"서버 오류: {e}")
            else:
                st.warning("이메일과 비밀번호를 모두 입력하세요.")

    job_options = ["영업직", "개발직", "디자이너", "기획", "고객상담"]
    keyword_options = ["출장이 잦은", "팀워크 중요", "고객 응대", "멀티태스킹", "책임감 요구"]

    selected_job = st.selectbox("희망 직무를 선택하세요", job_options)
    selected_keywords = st.multiselect("해당 직무의 특징 키워드를 선택하세요", keyword_options)

    if st.button("저장하기"):
        st.session_state["user_pref"] = {
            "job": selected_job,
            "keywords": selected_keywords
        }
        st.success("희망 직무 정보가 저장되었습니다.")


elif page == "🎤 면접 연습":
    if st.session_state['login_status']:

        st.title("🎤 면접 자기소개 연습")
        st.write("카메라, 타이머, 감정 분석을 활용해 실전처럼 연습해보세요.")

        # 카메라 실행
        ctx = webrtc_streamer(
            key="pitchpal-stream",
            video_transformer_factory=FaceEmotionAnalyzer,
            async_transform=True
        )

        # 타이머 로직
        duration_sec = 20  # 2분
        start_button = st.button("🎬 자기소개 시작")

        if "question_requested" not in st.session_state:
            st.session_state["question_requested"] = False

        if st.button("🎲 랜덤 질문 뽑기"):
            st.session_state["question_requested"] = True

        if "start_time" not in st.session_state:
            st.session_state.start_time = None
        if "recording" not in st.session_state:
            st.session_state.recording = False

        if start_button and ctx.video_transformer:
            st.session_state.start_time = time.time()
            ctx.video_transformer.recording = True
            st.session_state.recording = True
            st.success("자기소개 시작! 타이머가 작동 중입니다.")

            timer_placeholder = st.empty()
            while time.time() - st.session_state.start_time < duration_sec:
                remaining = duration_sec - int(time.time() - st.session_state.start_time)
                timer_placeholder.info(f"남은 시간: {remaining}초")
                time.sleep(1)

            # 종료
            ctx.video_transformer.recording = False
            st.session_state.recording = False
            st.success("⏰ 자기소개 종료!")

            # 감정 리포트 출력
            emotion_counts = Counter(ctx.video_transformer.emotion_log)
            st.subheader("📊 감정 분석 결과")
            # 감정 분석 결과 저장
            if emotion_counts:
                fig, ax = plt.subplots(figsize=(4, 4))  # 차트 크기 조정
                ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
                ax.set_title("말하는 동안 감정 분포")
                st.pyplot(fig)

                # GPT API로 퍼스널 면접 팁 요청
                with st.spinner("퍼스널 면접 팁 생성 중..."):
                    try:
                        prompt = f"""
아래는 면접 자기소개 연습 중 감정 분석 결과입니다.\n감정 분포: {dict(emotion_counts)}\n이 결과를 바탕으로, 지원자에게 도움이 될 만한 면접 팁을 2~3가지 제시해 주세요.\n구체적이고 실질적인 조언이면 좋겠습니다.
"""
                        gpt_resp = requests.post("http://localhost:8002/generate_questions", json={"prompt": prompt})
                        tips = gpt_resp.json().get("questions", "면접 팁 생성 실패")
                    except Exception as e:
                        tips = f"면접 팁 생성 오류: {e}"
                st.subheader("💡 퍼스널 면접 팁")
                st.write(tips)

                # 음성 녹음 UI 및 STT
                st.subheader("🎙️ 자기소개 음성 녹음")
                wav_audio_data = st_audiorec()
                stt_text = ""
                if wav_audio_data is not None:
                    st.audio(wav_audio_data, format='audio/wav')
                    st.info("음성 파일을 Whisper로 변환 중...")
                    # 파일을 임시로 저장 후 업로드
                    with open("temp_recorded.wav", "wb") as f:
                        f.write(wav_audio_data)
                    with open("temp_recorded.wav", "rb") as f:
                        files = {"file": ("recorded.wav", f, "audio/wav")}
                        try:
                            resp = requests.post("http://localhost:8002/stt", files=files)
                            if resp.status_code == 200:
                                stt_text = resp.json().get("text", "")
                                st.success("음성 → 텍스트 변환 결과:")
                                st.write(stt_text)
                            else:
                                st.error("STT 변환 실패: " + resp.text)
                        except Exception as e:
                            st.error(f"STT 서버 오류: {e}")

                # 저장할 기록 형태에 text도 추가
                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data": dict(emotion_counts),
                    "tips": tips,
                    "text": stt_text
                }

                # 세션 상태에 저장된 기록 리스트 초기화
                if "user_records" not in st.session_state:
                    st.session_state["user_records"] = {}

                # 현재 로그인한 사용자 이름 기준으로 저장
                if st.session_state['login_email'] not in st.session_state["user_records"]:
                    st.session_state["user_records"][st.session_state['login_email']] = []

                if st.button("💾 이 결과 저장하기"):
                    save_emotion_result(result)
                    st.success("감정 분석 결과가 저장되었습니다. 👤 마이페이지에서 확인할 수 있습니다.")
            else:
                st.warning("감정 분석 데이터가 부족합니다.")

            # 3️⃣ True일 때 질문 생성
            if not st.session_state.recording and st.session_state["question_requested"]:
                prefs = st.session_state.get("user_pref", {})
                job = prefs.get("job", "지원자")
                keywords = ", ".join(prefs.get("keywords", []))

                with st.spinner("질문 생성 중..."):
                    prompt = f"""
            너는 면접관이야. "{job}" 직무를 지원한 사람에게 "{keywords}" 키워드와 관련된 질문을 3개만 랜덤하게 생성해줘.
            면접처럼 자연스럽게 질문해줘. 질문 앞에 번호 붙여줘.
            """
                    try:
                        response = requests.post("http://localhost:8002/generate_questions", json={"prompt": prompt})
                        questions = response.json()["questions"]
                        st.subheader("🗣️ 랜덤 면접 질문")
                        st.write(questions)
                    except Exception as e:
                        st.error(f"❌ 질문 생성 실패: {e}")
    else:
        st.warning("로그인이 필요합니다.")


elif page == "✍️ 자기소개서 분석":
    if st.session_state['login_status']:
        st.title("✍️ 자기소개서 구조 및 역량 분석")
        user_input = st.text_area("자기소개서를 입력하세요", height=400)

        if st.button("🔍 분석하기"):
            if user_input.strip():
                with st.spinner("분석 중..."):
                    try:
                        response = requests.post("http://localhost:8002/analyze_resume", json={"text": user_input})
                        result = response.json()  # 이 줄에서 KeyError가 났다면, response.text를 보면 왜 그런지 알 수 있음

                        # 세션에 저장
                        st.session_state["analysis_result"] = result
                        st.session_state["last_resume_input"] = user_input

                        st.subheader("🧩 구조 분석")
                        st.write(result["structure"])

                    except Exception as e:
                        st.error(f"❌ 오류 발생: {e}")
        if st.session_state["analysis_result"]:
            result = st.session_state["analysis_result"]

            st.subheader("🧩 구조 분석")
            st.write(result["structure"])

            st.subheader("💼 역량 키워드 분석")
            st.write(result["keywords"])

            st.subheader("📝 종합 피드백")
            st.write(result["summary"])

            st.subheader("✏️ 개선 문장 예시")
            st.write(result["improvement"])

            # 💾 저장 버튼 (항상 렌더링됨)
            if st.button("💾 이 버전 저장하기"):
                if "resume_versions" not in st.session_state:
                    st.session_state["resume_versions"] = []

                st.session_state["resume_versions"].append({
                    "text": st.session_state["last_resume_input"],
                    "analysis": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("✅ 자기소개서 버전이 저장되었습니다.")
    else:
        st.warning("로그인 후 이용할 수 있습니다.")

elif page == "👤 마이페이지":

    st.title("📄 내 희망 직무 정보")

    prefs = st.session_state.get("user_pref")
    if prefs:
        st.write(f"**희망 직무:** {prefs['job']}")
        st.write(f"**중요 키워드:** {', '.join(prefs['keywords'])}")
    else:
        st.info("아직 희망 직무 정보를 입력하지 않았습니다.")

    st.title("👤 내 면접 연습 리포트")

    if st.session_state['login_status']:
        user_records = st.session_state.get("user_records", {}).get(st.session_state['login_email'], [])

        if user_records:
            for record in reversed(user_records):
                st.markdown(f"### 🕒 {record['timestamp']}")
                if 'text' in record and record['text']:
                    st.subheader("🗣️ 내가 말한 자기소개")
                    st.write(record['text'])
                st.write(record['data'])
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(record['data'].values(), labels=record['data'].keys(), autopct='%1.1f%%')
                st.pyplot(fig)
                if 'tips' in record:
                    st.subheader("💡 퍼스널 면접 팁")
                    st.write(record['tips'])
                st.divider()
        else:
            st.info("아직 저장된 리허설 기록이 없습니다.")
    else:
        st.warning("로그인 후 이용할 수 있습니다.")

    st.title("📄 저장된 자기소개서 버전")

    resume_versions = st.session_state.get("resume_versions", [])
    if resume_versions:
        for i, record in enumerate(resume_versions[::-1]):
            st.markdown(f"### 📝 버전 {len(resume_versions) - i} - {record['timestamp']}")
            st.code(record["text"])
            st.write(record["analysis"]["summary"])
            st.divider()
    else:
        st.info("저장된 자기소개서가 없습니다.")

elif page == "🔐 회원가입/비밀번호 찾기":
    st.title("🔐 회원가입 / 비밀번호 찾기")
    tabs = st.tabs(["회원가입", "비밀번호 찾기"])

    with tabs[0]:  # 회원가입
        st.subheader("회원가입")
        reg_email = st.text_input("이메일", key="reg_email")
        reg_pw = st.text_input("비밀번호", type="password", key="reg_pw")
        if st.button("회원가입", key="reg_btn"):
            if reg_email and reg_pw:
                try:
                    resp = requests.post("http://localhost:8002/register",
                                         json={"email": reg_email, "password": reg_pw})
                    if resp.status_code == 200:
                        st.success("회원가입이 완료되었습니다. 로그인 해주세요!")
                    else:
                        st.error(resp.json().get("detail", "회원가입 실패"))
                except Exception as e:
                    st.error(f"서버 오류: {e}")
            else:
                st.warning("이메일과 비밀번호를 모두 입력하세요.")

    with tabs[1]:  # 비밀번호 찾기
        st.subheader("비밀번호 찾기 (임시 비밀번호 발급)")
        reset_email = st.text_input("이메일", key="reset_email")
        if st.button("임시 비밀번호 발급", key="reset_btn"):
            if reset_email:
                try:
                    resp = requests.post("http://localhost:8002/reset_password", json={"email": reset_email})
                    if resp.status_code == 200:
                        temp_pw = resp.json().get("temp_password")
                        st.success(f"임시 비밀번호: {temp_pw}")
                    else:
                        st.error(resp.json().get("detail", "비밀번호 재설정 실패"))
                except Exception as e:
                    st.error(f"서버 오류: {e}")
            else:
                st.warning("이메일을 입력하세요.")


# 감정 분석 결과 저장 함수

def save_emotion_result(result):
    if "user_records" not in st.session_state:
        st.session_state["user_records"] = {}
    if st.session_state['login_email'] not in st.session_state["user_records"]:
        st.session_state["user_records"][st.session_state['login_email']] = []
    st.session_state["user_records"][st.session_state['login_email']].append(result)