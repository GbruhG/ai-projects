from __future__ import annotations
import os
from dotenv import load_dotenv
import uuid
from typing import List, Optional
import io
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    Form
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader

load_dotenv()

app = FastAPI(title="Job Assistant Bot Backend")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "http://127.0.0.1:8080", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionData:
    def __init__(self) -> None:
        self.general_memory = ConversationBufferMemory(return_messages=True)
        self.cover_letter_memory = ConversationBufferMemory(return_messages=True)


_session_store: dict[str, SessionData] = {}

SESSION_COOKIE_NAME = "job_assistant_session"


def get_session_id(request: Request, response: Response, session_id: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> str:  # type: ignore[name-match]

    if session_id and session_id in _session_store:
        return session_id

    new_id = str(uuid.uuid4())
    _session_store[new_id] = SessionData()
    response.set_cookie(key=SESSION_COOKIE_NAME, value=new_id, max_age=30 * 24 * 60 * 60, httponly=False)
    return new_id

def groq_chat() -> ChatGroq:

    return ChatGroq(model_name="llama3-70b-8192", temperature=0.3, max_tokens=1024)

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    history: List[dict]

@app.post("/chat/general", response_model=ChatResponse)
async def general_chat(req: ChatRequest, session_id: str = Depends(get_session_id)):

    session = _session_store[session_id]

    memory = session.general_memory
    memory.chat_memory.add_user_message(req.message)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful career coach giving concise and actionable job-hunting advice.",
        ),
        *memory.chat_memory.messages,
        ("human", "{user_message}"),
    ])

    chat = groq_chat()
    output = await chat.ainvoke(prompt.format_messages(user_message=req.message))

    memory.chat_memory.add_ai_message(output.content)

    return ChatResponse(
        response=output.content,
        history=[{"role": m.type, "content": m.content} for m in memory.chat_memory.messages],
    )


@app.post("/cover-letter", response_model=ChatResponse)
async def cover_letter_advisor(
    message: str = Form(..., description="User question or instructions"),
    file: Optional[UploadFile] = File(None),
    session_id: str = Depends(get_session_id),
):

    session = _session_store[session_id]
    memory = session.cover_letter_memory

    context = ""
    if file is not None:
        pdf_bytes = await file.read()
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            # Concatenate all pages
            context = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc

    memory.chat_memory.add_user_message(message)

    system_msg = (
        "You are an expert career consultant specialising in crafting persuasive cover letters."
        " Provide actionable advice or generate content as requested."
        " Try to not make it sound like an AI made it."
        " Keep track of the conversation and build upon previous discussions."
    )
    if context:
        system_msg += "\nHere is the user's existing cover letter for context: " + context[:4000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        *memory.chat_memory.messages,
        ("human", "{user_message}"),
    ])

    chat = groq_chat()
    output = await chat.ainvoke(prompt.format_messages(user_message=message))

    memory.chat_memory.add_ai_message(output.content)

    return ChatResponse(
        response=output.content,
        history=[{"role": m.type, "content": m.content} for m in memory.chat_memory.messages],
    )


@app.post("/cv-polish", response_model=ChatResponse)
async def cv_polisher(file: UploadFile = File(...)):

    pdf_bytes = await file.read()
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc

    system_prompt = (
        "You are a professional resume writer. Analyse the following CV and provide a polished version with improvements and recommendations."
    )
    chat = groq_chat()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=full_text[:4000])]
    output = await chat.ainvoke(messages)

    return ChatResponse(response=output.content, history=[{"role": "assistant", "content": output.content}])


@app.post("/clear-history")
async def clear_history(session_id: str = Depends(get_session_id)):

    _session_store[session_id].general_memory = ConversationBufferMemory(return_messages=True)
    return {"status": "cleared"}


@app.post("/clear-cover-letter-history")
async def clear_cover_letter_history(session_id: str = Depends(get_session_id)):

    _session_store[session_id].cover_letter_memory = ConversationBufferMemory(return_messages=True)
    return {"status": "cleared"}


@app.get("/")
async def root():
    return {"status": "Job Assistant Bot backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)