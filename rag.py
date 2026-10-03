from pathlib import Path
from typing import List

from dotenv import load_dotenv
import os
import certifi


# ==================================================
# Environment
# ==================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ==================================================
# LangChain / Chroma
# ==================================================

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# File readers
# ==================================================

from pypdf import PdfReader
import docx2txt


# ==================================================
# Directories
# ==================================================

Path("uploads").mkdir(
    exist_ok=True
)

Path("chroma_db").mkdir(
    exist_ok=True
)


# ==================================================
# Google Gemini Embeddings
# ==================================================

# IMPORTANT:
#
# This is an EMBEDDING model.
# It is NOT the chat model.
#
# Keep this as:
#
# gemini-embedding-001
#
# Your chat model is handled separately in agent.py
# using:
#
# gemini-3.8-flash
# ==================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ==================================================
# Chroma Vector Database
# ==================================================

vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# ==================================================
# Read uploaded files
# ==================================================

def read_file_text(file_path: str) -> str:

    path = Path(file_path)

    suffix = path.suffix.lower()

    # --------------------------------------------------
    # PDF
    # --------------------------------------------------

    if suffix == ".pdf":

        reader = PdfReader(
            file_path
        )

        text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

        return "\n".join(text)


    # --------------------------------------------------
    # DOCX
    # --------------------------------------------------

    if suffix == ".docx":

        return docx2txt.process(
            file_path
        )


    # --------------------------------------------------
    # Text / Markdown / Python / CSV
    # --------------------------------------------------

    if suffix in [
        ".txt",
        ".md",
        ".py",
        ".csv"
    ]:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )


    # --------------------------------------------------
    # Unsupported
    # --------------------------------------------------

    raise ValueError(
        "Unsupported file type. "
        "Upload PDF, DOCX, TXT, MD, PY, or CSV."
    )


# ==================================================
# Add document to RAG
# ==================================================

def add_document_to_rag(
    file_path: str,
    thread_id: str
):

    # --------------------------------------------------
    # Read document
    # --------------------------------------------------

    text = read_file_text(
        file_path
    )

    if not text.strip():

        raise ValueError(
            "No text could be extracted from this file."
        )


    # --------------------------------------------------
    # Split document into chunks
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(
        text
    )


    # --------------------------------------------------
    # Create LangChain documents
    # --------------------------------------------------

    docs: List[Document] = []

    source_name = Path(
        file_path
    ).name

    for chunk in chunks:

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "thread_id": thread_id,
                    "source": source_name
                }
            )
        )


    # --------------------------------------------------
    # Store embeddings in Chroma
    # --------------------------------------------------

    if docs:

        vectorstore.add_documents(
            docs
        )


    # --------------------------------------------------
    # Return upload information
    # --------------------------------------------------

    return {
        "filename": source_name,
        "chunks": len(docs)
    }


# ==================================================
# Retrieve relevant document content
# ==================================================

def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4
) -> str:

    # --------------------------------------------------
    # Search only this conversation's documents
    # --------------------------------------------------

    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={
            "thread_id": thread_id
        }
    )


    # --------------------------------------------------
    # No results
    # --------------------------------------------------

    if not docs:

        return (
            "No relevant uploaded document "
            "content found."
        )


    # --------------------------------------------------
    # Format results
    # --------------------------------------------------

    results = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "uploaded document"
        )

        results.append(
            f"[Source {i}: {source}]\n"
            f"{doc.page_content}"
        )


    return "\n\n".join(
        results
    )