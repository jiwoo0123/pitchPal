from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")

labels = ["부정", "긍정"]

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1)

    sentiment = labels[pred.item()]
    score = round(probs[0][pred.item()].item(), 2)

    return {
        "sentiment": sentiment,
        "score": score
    }
