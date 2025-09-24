from fastapi import FastAPI, Body
from transformers import pipeline

app = FastAPI(title="Laptop Assistant (GPT-2)")

generator = pipeline("text-generation", model="gpt2")

SYSTEM_PROMPT = """
You are Laptop Assistant.
Answer laptop questions, compare models, and give concise recommendations.
You are a helpful assistant specialized in laptops and notebook computers.
You can answer questions about laptop specifications, compare different models, suggest laptops for specific needs, and provide concise, clear recommendations.
If you need more information from the user, ask clarifying questions.
Keep your answers factual and avoid speculation.

"""

@app.post("/chat")
def chat(user_input: str = Body(..., embed=True)):
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
    response = generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    return {"user": user_input, "assistant": response[0]["generated_text"]}
