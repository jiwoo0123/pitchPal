import requests

res = requests.post("http://localhost:8001/analyze_resume", json={"text": "이것은 테스트 자기소개서입니다."})
print("응답 코드:", res.status_code)
print("응답 내용:", res.text)
