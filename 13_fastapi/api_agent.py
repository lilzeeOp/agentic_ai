"""
STEP 13: FastAPI Agent — Your Agent as a Web API
==================================================

WHAT YOU'LL LEARN:
- How to turn an agent into a web API using FastAPI
- How endpoints work (POST /chat, GET /)
- How request/response bodies work (Pydantic BaseModel)
- How any app, website, or program can talk to your agent

HOW TO RUN:
  uvicorn api_agent:app --reload

Then open http://localhost:8000/docs to test it.

TODO (when you learn more Python):
- Add streaming endpoint with StreamingResponse
- Add tool-calling agent endpoint
- Add conversation history support
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

app = FastAPI(title="AI Agent API")


# This defines what the request body looks like
class ChatRequest(BaseModel):
    message: str


# ENDPOINT 1: Simple chat
@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": request.message}],
    )
    return {"reply": response.choices[0].message.content}


# ENDPOINT 2: List available endpoints
@app.get("/")
def home():
    return {
        "message": "AI Agent API is running",
        "endpoints": ["/chat", "/docs"]
    }
