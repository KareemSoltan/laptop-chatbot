import requests, os, time
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/laptops")


sessions = {}


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


def fetch_laptops():
    try:
        response = requests.get(BACKEND_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching laptops: {e}")
    return []


def call_hf_api(payload, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"HF API error: {e}")
        time.sleep(delay)
    return None


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id
    history = sessions.get(session_id, [])

   
    laptops = fetch_laptops()

    # Build context
    context = "You are a helpful laptop sales assistant.\n"
    context += "Only recommend laptops from the list below:\n"
    for laptop in laptops:
        context += f"- {laptop['brand']} {laptop['model']} (${laptop['price']}): {laptop['specs']}\n"

    for turn in history:
        context += f"{turn['role'].capitalize()}: {turn['content']}\n"

    context += f"Customer: {req.user_message}\nAssistant:"

    
    payload = {"inputs": context, "parameters": {"max_new_tokens": 200}}
    data = call_hf_api(payload)

    
    if data and isinstance(data, list) and "generated_text" in data[0]:
        reply = data[0]["generated_text"].split("Assistant:")[-1].strip()
    else:
        reply = "Sorry, I’m having trouble right now. Please try again later."

    
    history.append({"role": "user", "content": req.user_message})
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history

    
    return {"response": reply}

