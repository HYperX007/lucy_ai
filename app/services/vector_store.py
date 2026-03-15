import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger("L.U.C.Y.")

class VectorStoreService:
    """
    Builds a FAISS index from learning_data .txt files and chats_data .json files,
    and provides a retriever to fetch the k most relevant chunks for a query.
    """

    def __init__(self):
        """Initialize embeddings, text splitter, and cache."""
        # Define attributes first to avoid AttributeErrors
        self.vector_store: Optional[FAISS] = None
        self._retriever_cache = {} 

        # Load local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    # --------------------------------------------------------------------------
    # LOAD DOCUMENTS FROM DISK
    # --------------------------------------------------------------------------

    def load_learning_data(self) -> List[Document]:
        """Read all .txt files in database/learning_data/."""
        documents = []
        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": str(file_path.name)}))
                        logger.info("[VECTOR] Loaded learning data: %s", file_path.name)
            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)
        return documents

    def load_chat_history(self) -> List[Document]:
        """Load past chat JSON files and format for retrieval."""
        documents = []
        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                messages = chat_data.get("messages", [])
                chat_content = "\n".join([
                    f"User: {msg.get('content', '')}" if msg.get('role') == 'user'
                    else f"Assistant: {msg.get('content', '')}"
                    for msg in messages
                ])
                if chat_content.strip():
                    documents.append(Document(page_content=chat_content, metadata={"source": f"chat_{file_path.stem}"}))
            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)
        return documents

    # --------------------------------------------------------------------------
    # BUILD AND SAVE FAISS INDEX
    # --------------------------------------------------------------------------

    def create_vector_store(self) -> FAISS:
        """Load data, chunk, embed, build FAISS, and save to disk."""
        # Clear existing cache whenever the store is rebuilt
        self._retriever_cache.clear()
        
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()
        all_documents = learning_docs + chat_docs

        if not all_documents:
            self.vector_store = FAISS.from_texts(["No data available yet."], self.embeddings)
            logger.info("[VECTOR] Created placeholder index")
        else:
            chunks = self.text_splitter.split_documents(all_documents)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("[VECTOR] FAISS index built with %d vectors", len(chunks))
            
        self.save_vector_store()
        return self.vector_store

    def save_vector_store(self):
        """Write the current FAISS index to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(str(VECTOR_STORE_DIR))
                logger.info("[VECTOR] Vector store saved to disk.")
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    # --------------------------------------------------------------------------
    # RETRIEVER FOR CONTEXT
    # --------------------------------------------------------------------------

    def get_retriever(self, k: int = 10):
        """Return a cached retriever for the FAISS index."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")
        
        if k not in self._retriever_cache:
            self._retriever_cache[k] = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        return self._retriever_cache[k]