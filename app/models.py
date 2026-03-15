"""
DATA MODELS MODULE
==================

This file defines the Pydantic models used for API requests, responses, and
internal chat storage. FastAPI uses these to validate incoming JSON and to
serialize responses; the chat service uses them when saving/loading sessions.

MODELS:
    ChatRequest  - Body of POST /chat and POST /chat/realtime (message + optional session_id).
    ChatResponse - Body returned by both chat endpoints (response text + session_id).
    ChatMessage  - One message in a conversation (role + content). Used inside ChatHistory.
    ChatHistory  - Full conversation: session_id + list of ChatMessage. Used when saving to disk.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# -----------------------------------------------------------------------------
# MESSAGE AND REQUEST/RESPONSE MODELS
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):

    role: str    # Either "user" (human) or "assistant" (Lucy).
    content: str # The message text.


class ChatRequest(BaseModel):

    # ... means required; min/max_length prevent empty input and token overflow.
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[str] = None
    tts: bool = False
    
    
class ChatResponse(BaseModel):

    response: str
    session_id: str


class ChatHistory(BaseModel):

    session_id: str
    messages: List[ChatMessage]
    
    
class TTSRequest(BaseModel):
    
    text: str = Field(..., min_length=1, max_length=5000 )