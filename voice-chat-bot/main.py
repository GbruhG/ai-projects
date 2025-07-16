from fastapi import FastAPI, File, UploadFile, Response, Cookie, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import logging
import json
import uuid

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Knowledge Graph Query API", version="1.0.0")

chat_history = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IndexResponse(BaseModel):
    message: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

@app.post("/upload-audio/", response_model=QueryResponse)
async def upload_audio(file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    query_text = speech_to_text(file)
    response_text = process_query(query_text, None, user_id)
    return QueryResponse(query=query_text, response=response_text)

@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, user_id: str = Depends(get_user_id)):
    response_text = process_query(payload.query, None, user_id)
    return QueryResponse(query=payload.query, response=response_text)

@app.get("/get-user-id/")
def get_user_id(user_id: str | None = Cookie(default=None), response: Response | None = None):
    if not user_id:
        user_id = str(uuid.uuid4())
        if response is not None:
            response.set_cookie(key="user_id", value=user_id)
    return user_id

def add_to_history(user_id, prompt, response):
    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append((prompt, response))

def process_query(prompt, system_prompt=None, user_id=None):
    if system_prompt is not None:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
    else:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
    if user_id is None:
        return {"error": "No user_id cookie found"}
    
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    response = response.choices[0].message.content
    chat_history[user_id].append((prompt, response))
    
    return response

def speech_to_text(speech):
    transcription = client.audio.transcriptions.create(
      file=speech, # Required audio file
      model="whisper-large-v3-turbo", # Required model to use for transcription
      prompt="Specify context or spelling",  # Optional
      response_format="verbose_json",  # Optional
      timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
      language="en",  # Optional
      temperature=0.0  # Optional
    )
    # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
    return transcription.text
    
def text_to_speech(text):
    response = client.audio.speech.create(
        model="playai-tts",
        voice="Fritz-PlayAI",
        input=text,
        response_format="wav"
    )

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)