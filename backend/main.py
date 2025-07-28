from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import openai
import backend.users as users
from fastapi import HTTPException
from backend import speech

openai.api_key = "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"

app = FastAPI()

# CORS 설정 (Streamlit 연동 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 시 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(speech.router)

class PromptRequest(BaseModel):
    prompt: str
class ResumeRequest(BaseModel):
    text: str

class RegisterRequest(BaseModel):
    email: str
    password: str

class ResetPasswordRequest(BaseModel):
    email: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/register")
async def register(req: RegisterRequest):
    if users.add_user(req.email, req.password):
        return {"success": True, "message": "회원가입이 완료되었습니다."}
    else:
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")

@app.post("/reset_password")
async def reset_password(req: ResetPasswordRequest):
    user = users.get_user(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="존재하지 않는 이메일입니다.")
    temp_pw = users.generate_temp_password()
    users.update_password(req.email, temp_pw)
    return {"success": True, "temp_password": temp_pw, "message": "임시 비밀번호가 발급되었습니다."}

@app.post("/login")
async def login(req: LoginRequest):
    user = users.get_user(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="존재하지 않는 이메일입니다.")
    if user["password"] != req.password:
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")
    return {"success": True, "message": "로그인 성공", "email": req.email}


@app.post("/analyze_resume")
async def analyze_resume(request: ResumeRequest):
    print("🔥 analyze_resume 호출됨")
    text = request.text

    structure_feedback = analyze_structure(text)
    keyword_feedback = analyze_keywords_and_experiences(text)
    summary_feedback = summarize_feedback(structure_feedback, keyword_feedback)
    improved_examples = generate_improved_examples(text)

    return {
        "structure": structure_feedback,
        "keywords": keyword_feedback,
        "summary": summary_feedback,
        "improvement": improved_examples
    }

def call_openai(prompt: str) -> str:
    print("📤 GPT 호출 시작됨")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 인사담당자이며 자기소개서를 분석하는 전문가야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        print("📥 GPT 응답 완료")
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("🔥 OpenAI API 호출 중 오류 발생:", repr(e))  # ← 여기에 repr 추가해서 로그를 꼭 찍자
        return f"❌ OpenAI API 호출 중 오류가 발생했습니다: {str(e)}"

def call_openai_dynamic(system_role: str, user_prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("🔥 GPT 호출 오류:", repr(e))
        return f"❌ OpenAI 오류: {str(e)}"


@app.post("/generate_questions")
async def generate_questions(req: PromptRequest):
    try:
        result = call_openai_dynamic(
            system_role="너는 면접관이야. 지원자에게 직무와 관련된 면접 질문을 랜덤하게 3개 생성하는 역할이야.",
            user_prompt=req.prompt
        )
        return {"questions": result}
    except Exception as e:
        return {"questions": f"❌ 오류 발생: {str(e)}"}
def generate_improved_examples(text: str) -> str:
    prompt = f"""
다음 자기소개서를 개선하기 위한 문장을 2~3개 제안해주세요.

자기소개서:
{text}

→ 개선된 문장은 실제 문장 형태로 제공해주세요.
"""
    return call_openai(prompt)

def analyze_structure(text: str) -> str:
    prompt = f"""
다음 자기소개서의 문단 구성과 흐름을 분석해주세요.

1. 도입 → 전개 → 결론 구조가 명확한가요?
2. 각 문단은 논리적으로 연결되어 있나요?
3. 주제와 메시지가 선명하게 드러나나요?

자기소개서:
{text}

→ 분석 결과를 구조적으로 정리해주세요.
"""
    return call_openai(prompt)

def analyze_keywords_and_experiences(text: str) -> str:
    prompt = f"""
다음 자기소개서에서 지원자가 강조하는 주요 역량 키워드를 추출하고, 해당 키워드를 뒷받침하는 경험이 충분히 서술되었는지 평가해주세요.

1. 주요 키워드 (예: 책임감, 문제해결 등)
2. 각 키워드에 연결된 사례 유무
3. 부족하거나 과장된 표현은 없는지

자기소개서:
{text}

→ 분석 결과를 항목별로 정리해주세요.
"""
    return call_openai(prompt)

def summarize_feedback(structure: str, keywords: str) -> str:
    prompt = f"""
다음은 자기소개서의 분석 결과입니다.

[구조 분석]
{structure}

[역량 분석]
{keywords}

이 두 가지 분석 결과를 종합해서, 면접관의 시선으로 종합 피드백을 작성해주세요.
"""
    return call_openai(prompt)