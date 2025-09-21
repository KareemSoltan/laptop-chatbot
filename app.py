from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

app = FastAPI(title="Simple Chatbot API")

class ChatRequest(BaseModel):
    message: str

chat_history = []

def generate_response(user_input: str, max_length=100):
    chat_history.append(f"User: {user_input}")
    input_text = "\n".join(chat_history) + "\nBot:"
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = response_text.split("Bot:")[-1].strip()
    
    chat_history.append(f"Bot: {bot_reply}")
    return bot_reply

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        reply = generate_response(req.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Simple Chatbot API!"}
