"""
CHAT SERVICE MODULE
====================

This owns all the chat session and conversation logic. It is used by the
/chat and /chat/realtime endpoints. Designed for single-user use: one server
has one ChatService and one in-memory session store; the user can have many 
sessions (each identifided by a session_id).

RESPONSIBILITIES:
  - get_or_create_session(session_id): Return existing session or create a new one.
    If the user sends a session_id that was used before (e.g. before a restart),
    we try to load it from disk so the converstation continues.
  - add_message / get_chat_history: keep messages in memory per session.
  - format history for llm: Turn the message list into (user, assiatnce) pairs
    and trim to MAX_CHAT_HISTORY_TURNS so we dont overflow the prompt.
  - process_message / process_realtime_message: Add user message, call Groq (or 
    RealtimeGroq), add assistant reply, return reply.
  - save_chat_session: write session to database/chats_data/*.json so it persists 
    and can be loaded on next startup (and used by the vector store for retrieval).
"""

import json
import logging
import time
from pathlib import Path 
from typing import Any, Iterator, List, Dict, Optional
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService


logger = logging.getLogger("L.U.C.Y")


SAVE_EVERY_N_CHUNKS = 5  # how often to save the session to disk (every N messages) to balance durability with performance.c

#===========================================================================
# CHAT SERVICE CLASS
#===========================================================================

class ChatService: 
    """
    Manages chat sessions: in memory manage lists, load/save to disk, and
    calling Groq (or realtime) to get replies. All state for active sessions
    is in self.session; saving to disk is done after each message so 
    conversations survive restarts.
    """
    
    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """Store reference to the GroqService and RealtimeGroqService, keep sessions in memory."""
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        # Map session_id -> list of ChatMessage objects (user and assistant messages in order)
        self.sessions: Dict[str, List[ChatMessage]] = {}
        
    # --------------------------------------------------------------------
    # SESSION LOAD / VALIDATE / GET-OR-CREATE
    # --------------------------------------------------------------------
    
    def load_session_from_disk(self, session_id: str) -> bool:
        """
        load a session from database/chats_data/ if a file for this session_id exists.
        
        File name is chat_{safe_session_id}.json where safe_session_id has dashes/spaces removed.
        On success we put the messages into self.sessions[session_id] so later requests use them.
        Returns True if loaded, False if file missing or unreadable.
        """
        # Sanitize id for use as filename (remove dashes and spaces)
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        file_path = CHATS_DATA_DIR / filename
        
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
              # convert stored dicts back to ChatMessage objects.
            messages = [
                ChatMessage(role=msg.get("role"), content=msg.get("content"))
                for msg in chat_dict.get("messages", [])
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning(f"Failed to load session {session_id} from disk: {e}")
            return False
      
    def validate_session_id(self, session_id: str) -> bool:
        """
        Return true if session_id is safe to use (not empty, no path traversal, length <= 255).
        Used to reject malicious or invalid IDs before we use them in file paths.
        """
        if not session_id or not session_id.strip():
            return False
        # Block path traveral or path seperators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True
        
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
            """
            Return a session_id and ensure that session exsists in memory
            
            - If session_id is None: create a new sessiom with a new UUID and return that.
            - If session_id is provided: validate it, if its in self.sessions return it;
              else try to load it from disk; if not found, create a new session with that ID.
            Raises ValueError if session_id is invalid (empty, path traversal, too long).
            """
            t0=time.perf_counter()
            if not session_id:
                new_session_id = str(uuid.uuid4())
                self.sessions[new_session_id] = []
                logger.info("[TIMING] session_get_or_create: %.3fs (new)",time.perf_counter()-t0)
                return new_session_id
              
            if not self.validate_session_id(session_id):
              raise ValueError(
                f"Invalid session_id format: {session_id}. Session Id must be non-empty," 
                "not contain path traversal characters, and be under 255 characters."
                )
              
            if session_id in self.sessions:
                logger.info("[TIMING] session_get_or_create: %.3fs (memory)",time.perf_counter()-t0)
                return session_id
            
            if self.load_session_from_disk(session_id):
                logger.info("[TIMING] session_get_or_create: %.3fs (disk)",time.perf_counter()-t0)
                return session_id
              
            # New session with this ID (e.g. client sent an ID that was never saved)
            self.sessions[session_id] = []
            logger.info("[TIMING] session_get_or_create: %.3fs (new_id)",time.perf_counter()-t0)
            return session_id
          
    # --------------------------------------------------------------------
    # MESSAGES AND HISTORY FORMATTING
    # --------------------------------------------------------------------
    
    def add_message(self, session_id: str, role: str, content: str):
        """Append one message (user or assistant) to the session's message list. Create session if missing."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))
        
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
      """Return the list of messages for this session (chronological). Empty list if session unknown."""
      return self.sessions.get(session_id, [])
    
    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[tuple]:
      """
      Build a list of (user_text, assistant_text) pairs for the LLM prompt.
      
      we only include complete pairs and cap at MAX_CHAT_HISTORY_TURNS (e.g. 20).
      so the prompt does not grow unbounded. If exclude_last is True we drop the
      last message (the current user message that we are about to reply to).
      """
      messages = self.get_chat_history(session_id)
      history = []
      # if exclude_last, we skip the last message (the current user message we are about to reply to).
      messages_to_process = messages[:-1] if exclude_last and messages else messages
      i = 0
      while i < len(messages_to_process) - 1:
          user_msg = messages_to_process[i]
          ai_msg = messages_to_process[i + 1]
          if user_msg.role == "user" and ai_msg.role == "assistant":
              history.append((user_msg.content, ai_msg.content))
              i += 2
          else:
              i += 1
      # Keep only the most recent turns so that the prompt does not exceed token limits.
      if len(history) > MAX_CHAT_HISTORY_TURNS:
          history = history[-MAX_CHAT_HISTORY_TURNS:]
      return history
        
    # --------------------------------------------------------------------
    # PROCESS MESSAGE (REALTIME AND GENERAL) 
    # --------------------------------------------------------------------
    
    def process_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one general chat message: add user message, call Groq (no web search), add reply, return reply.
        """
        logger.info("[GENERAL] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL] History pairs sent to LLM: %d", len(chat_history))
        response = self.groq_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        logger.info("[GENERAL] Response length: %d chars | Preview: %.120s", len(response), response)
        self.save_chat_session(session_id)
        return response
      
    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one realtime chat message: add user message, call realtime service (Tavily + Groq), add reply, return it.
        Uses the same session as process_message so the history is shared. Raises ValueError if realtime_service is None.
        """
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized, cannot process realtime queries.")
        logger.info("[REALTIME] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[REALTIME] History pairs sent to Realtime LLM: %d", len(chat_history))
        response = self.realtime_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        logger.info("[REALTIME] Response length: %d chars | Preview: %.120s", len(response), response)
        self.save_chat_session(session_id)
        return response
      
    def process_message_stream(self, session_id: str, user_message: str) -> Iterator[str]:
        logger.info("[GENERAL-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")  # Placeholder for the assistant message that we will build up.
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)   
        logger.info("[GENERAL-STREAM] History pairs sent to LLM: %d", len(chat_history))
        chunk_count = 0
        try:
            for chunk in self.groq_service.stream_response(question=user_message, chat_history=chat_history):
                # Update the last assistant message with the new chunk.
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1
                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    self.save_chat_session(session_id, log_timing=False)
                yield chunk  # Yield the chunk to the client after saving.
        finally:
            final_response = self.sessions[session_id][-1].content
            logger.info("[GENERAL-STREAM] Completed | Chunk: %d | Response length: %d chars", chunk_count, len(final_response))
            self.save_chat_session(session_id)
            
    def process_realtime_message_stream(self, session_id: str, user_message: str) -> Iterator[Any]:
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")
            
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "") 
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        
        chunk_count = 0
        try:
            for chunk in self.realtime_service.stream_response(question=user_message, chat_history=chat_history):
                # IMPORTANT: Pass the dictionary chunk to main.py to trigger the side panel
                if isinstance(chunk, dict):
                    yield chunk
                elif isinstance(chunk, str):
                    self.sessions[session_id][-1].content += chunk
                    chunk_count += 1
                    if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                        self.save_chat_session(session_id, log_timing=False)
                    yield chunk 
        finally:
            self.save_chat_session(session_id)
    
    # --------------------------------------------------------------------
    # PERSIST SESSION TO DISK
    # --------------------------------------------------------------------
      
    def save_chat_session(self, session_id: str, log_timing: bool = True):
      """
      write this session's messages to database/chats_data/chat_{safe_id}.json.
        
      called after each message to the conversation is persisted. The vector store
      is rebuild on startup from these files, so now chats are included after restart.
      If the session is missing or reply we do nothing. On write error we only log.
      """
      if session_id not in self.sessions:
          return
        
      messages = self.sessions[session_id]
      safe_session_id = session_id.replace("-", "").replace(" ", "")
      filename = f"chat_{safe_session_id}.json"
      file_path = CHATS_DATA_DIR / filename
      chat_dict = {
          "session_id": session_id,
          "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
      }
        
      try:
          t0 = time.perf_counter() if log_timing else 0
          with open(file_path, "w", encoding="utf-8") as f:
              json.dump(chat_dict, f, ensure_ascii=False, indent=2)
          if log_timing:
            logger.info("[TIMING] Save_session_json: %s | %.3fs ", session_id[:12], time.perf_counter() - t0)
      except Exception as e:
          logger.error("Failed to save chat session %s to disk: %s", session_id, e)