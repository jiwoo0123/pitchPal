from fastapi import APIRouter, UploadFile, File, FastAPI
import whisper
import os

router = APIRouter()

model = None

def get_whisper_model():
    global model
    if model is None:
        model = whisper.load_model("base")
    return model

@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    model = get_whisper_model()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    result = model.transcribe(temp_path, language="ko")
    os.remove(temp_path)
    return {"text": result["text"]}

app = FastAPI()
app.include_router(router)