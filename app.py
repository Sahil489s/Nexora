from dotenv import load_dotenv
import os
import certifi

# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# --------------------------------------------------
# Standard imports
# --------------------------------------------------

import json
import uuid
from pathlib import Path


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

import uvicorn

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form
)

from fastapi.responses import (
    StreamingResponse,
    JSONResponse
)

from fastapi.templating import Jinja2Templates


# --------------------------------------------------
# LangChain messages
# --------------------------------------------------

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from agent import get_agent

from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations
)

from rag import add_document_to_rag

from tools import set_current_thread_id


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI()


templates = Jinja2Templates(
    directory="templates"
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

Path("uploads").mkdir(
    exist_ok=True
)

Path("data").mkdir(
    exist_ok=True
)


# --------------------------------------------------
# Database initialization
# --------------------------------------------------

init_db()


# ==================================================
# HOME
# ==================================================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ==================================================
# CONVERSATIONS
# ==================================================

@app.get("/conversations")
async def conversations():

    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


# ==================================================
# CHAT HISTORY
# ==================================================

@app.get("/history/{thread_id}")
async def history(thread_id: str):

    messages = get_chat_history(
        thread_id
    )

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


# ==================================================
# DOCUMENT UPLOAD
# ==================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):

    try:

        allowed_extensions = [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".csv"
        ]

        filename = file.filename or "uploaded_file"

        suffix = Path(
            filename
        ).suffix.lower()

        if suffix not in allowed_extensions:

            return JSONResponse(
                {
                    "success": False,
                    "message": (
                        "Unsupported file type. "
                        "Upload PDF, DOCX, TXT, MD, PY, or CSV."
                    )
                },
                status_code=400
            )

        # ------------------------------------------
        # Generate unique file ID
        # ------------------------------------------

        file_id = str(
            uuid.uuid4()
        )

        safe_filename = filename.replace(
            " ",
            "_"
        )

        file_path = (
            f"uploads/{file_id}_{safe_filename}"
        )

        # ------------------------------------------
        # Save file
        # ------------------------------------------

        file_content = await file.read()

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                file_content
            )

        # ------------------------------------------
        # Update conversation
        # ------------------------------------------

        create_or_update_conversation(
            thread_id,
            "Uploaded document"
        )

        # ------------------------------------------
        # Add document to RAG
        # ------------------------------------------

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse(
            {
                "success": True,
                "message": (
                    f"Uploaded {result['filename']} "
                    f"and created {result['chunks']} chunks."
                )
            }
        )

    except Exception as e:

        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )


# ==================================================
# SERVER-SENT EVENTS
# ==================================================

def sse_data(payload: dict) -> str:

    return (
        f"data: "
        f"{json.dumps(payload, ensure_ascii=False)}"
        f"\n\n"
    )


# ==================================================
# STREAM FILTER
# ==================================================

def should_stream_chunk(
    chunk,
    metadata
) -> bool:

    """
    Prevent raw tool/search/RAG output
    from appearing in the frontend.

    Only normal AI text is streamed.
    """

    metadata = metadata or {}

    node_name = str(
        metadata.get(
            "langgraph_node",
            ""
        )
    ).lower()

    # ------------------------------------------
    # Ignore tool nodes
    # ------------------------------------------

    if "tool" in node_name:
        return False

    # ------------------------------------------
    # Ignore ToolMessage
    # ------------------------------------------

    if isinstance(
        chunk,
        ToolMessage
    ):
        return False

    # ------------------------------------------
    # Only AI messages
    # ------------------------------------------

    if not isinstance(
        chunk,
        (
            AIMessage,
            AIMessageChunk
        )
    ):
        return False

    # ------------------------------------------
    # Ignore tool calls
    # ------------------------------------------

    if getattr(
        chunk,
        "tool_calls",
        None
    ):
        return False

    if getattr(
        chunk,
        "invalid_tool_calls",
        None
    ):
        return False

    # ------------------------------------------
    # Ignore additional tool calls
    # ------------------------------------------

    additional_kwargs = getattr(
        chunk,
        "additional_kwargs",
        {}
    ) or {}

    if additional_kwargs.get(
        "tool_calls"
    ):
        return False

    return True


# ==================================================
# EXTRACT TEXT
# ==================================================

def extract_text_from_chunk(
    chunk
) -> str:

    content = getattr(
        chunk,
        "content",
        ""
    )

    if not content:
        return ""

    # ------------------------------------------
    # Normal string
    # ------------------------------------------

    if isinstance(
        content,
        str
    ):
        return content

    # ------------------------------------------
    # List content
    # ------------------------------------------

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                str
            ):

                text_parts.append(
                    item
                )

            elif isinstance(
                item,
                dict
            ):

                if (
                    item.get("type") == "text"
                    and isinstance(
                        item.get("text"),
                        str
                    )
                ):

                    text_parts.append(
                        item["text"]
                    )

                elif isinstance(
                    item.get("text"),
                    str
                ):

                    text_parts.append(
                        item["text"]
                    )

                elif isinstance(
                    item.get("content"),
                    str
                ):

                    text_parts.append(
                        item["content"]
                    )

        return "".join(
            text_parts
        )

    return ""


# ==================================================
# CHAT STREAM
# ==================================================

@app.post("/chat/stream")
async def chat_stream(
    request: Request
):

    # ------------------------------------------
    # Read request
    # ------------------------------------------

    try:

        data = await request.json()

    except Exception:

        return JSONResponse(
            {
                "error": "Invalid JSON body."
            },
            status_code=400
        )

    # ------------------------------------------
    # Request data
    # ------------------------------------------

    user_message = data.get(
        "message",
        ""
    )

    thread_id = data.get(
        "thread_id",
        "default"
    )

    # IMPORTANT:
    # Gemini 3.8 Flash is now the default.
    selected_model = data.get(
        "model",
        "gemini-3.8-flash"
    )

    # ------------------------------------------
    # Validate message
    # ------------------------------------------

    if not user_message.strip():

        return JSONResponse(
            {
                "error": "Message is required."
            },
            status_code=400
        )

    # ------------------------------------------
    # Create agent
    # ------------------------------------------

    try:

        agent = get_agent(
            selected_model
        )

    except Exception as e:

        return JSONResponse(
            {
                "error": (
                    f"Failed to initialize Gemini model: "
                    f"{str(e)}"
                )
            },
            status_code=500
        )

    # ------------------------------------------
    # Save user message
    # ------------------------------------------

    create_or_update_conversation(
        thread_id,
        user_message
    )

    save_chat_message(
        thread_id,
        "user",
        user_message
    )

    # ------------------------------------------
    # Set current thread
    # ------------------------------------------

    set_current_thread_id(
        thread_id
    )

    # ------------------------------------------
    # LangGraph config
    # ------------------------------------------

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ==================================================
    # SSE EVENT GENERATOR
    # ==================================================

    def event_generator():

        final_answer = ""

        try:

            inputs = {
                "messages": [
                    HumanMessage(
                        content=user_message
                    )
                ]
            }

            # ------------------------------------------
            # Stream LangGraph response
            # ------------------------------------------

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):

                # --------------------------------------
                # Filter tool output
                # --------------------------------------

                if not should_stream_chunk(
                    chunk,
                    metadata
                ):
                    continue

                # --------------------------------------
                # Extract text
                # --------------------------------------

                token = extract_text_from_chunk(
                    chunk
                )

                if token:

                    final_answer += token

                    yield sse_data(
                        {
                            "token": token
                        }
                    )

            # ------------------------------------------
            # Save assistant response
            # ------------------------------------------

            if final_answer.strip():

                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer
                )

            # ------------------------------------------
            # Done
            # ------------------------------------------

            yield sse_data(
                {
                    "done": True
                }
            )

        except Exception as e:

            # ------------------------------------------
            # Send error to frontend
            # ------------------------------------------

            yield sse_data(
                {
                    "error": str(e)
                }
            )

            yield sse_data(
                {
                    "done": True
                }
            )

    # ==================================================
    # STREAM RESPONSE
    # ==================================================

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )