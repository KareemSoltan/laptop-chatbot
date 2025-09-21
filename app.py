import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "google/flan-t5-small"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

app = FastAPI()
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

def call_hf_api(prompt, retries=3, delay=2):
    payload = {"inputs": prompt}
    for attempt in range(retries):
        try:
            response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
            else:
                print(f"HF API error: {response.status_code} {response.text}")
        except Exception as e:
            print(f"HF API exception: {e}")
        import time
        time.sleep(delay)
    return "Sorry, I’m having trouble right now. Please try again later."

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    history = sessions.get(session_id, [])
    if len(history) > 6:
        history = history[-6:]
    prompt = f"You are a helpful laptop sales assistant. Answer this question:\n{req.user_message}"
    reply = call_hf_api(prompt)
    history.append({"role": "user", "content": req.user_message})
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history
    return {"response": reply}
