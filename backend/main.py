"""
RLM FastAPI Backend
===================

REST API for the Recursive Language Model application.
Supports:
- Document upload (PDF, TXT, DOCX, MD, code files)
- Querying with RLM retrieval
- Session management
"""

import os
import uuid
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import aiofiles

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.rlm_engine import RLMEngine, RLMEngineNoSubCalls
from core.llm_client import LLMClientFactory
from core.document_processor import DocumentProcessor, Document

# Configuration
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory session storage (use Redis in production)
sessions: Dict[str, Dict[str, Any]] = {}

# Global document processor
doc_processor = DocumentProcessor(chunk_size=100000, chunk_overlap=1000)

# Pydantic models for API
class QueryRequest(BaseModel):
    session_id: str
    query: str
    use_subcalls: bool = True
    max_iterations: int = 50

class QueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    iterations: int = 0
    sub_lm_calls: int = 0
    processing_time_seconds: float = 0.0

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    documents: List[Dict[str, Any]]
    total_chars: int

class DocumentInfo(BaseModel):
    filename: str
    doc_type: str
    total_chars: int
    num_chunks: int
    metadata: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("=" * 60)
    print("RLM Application Starting")
    print("=" * 60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Chunk size: {doc_processor.chunk_size}")
    print("=" * 60)
    yield
    # Shutdown
    print("Shutting down RLM Application")


app = FastAPI(
    title="RLM - Recursive Language Model",
    description="Process arbitrarily long documents using recursive LLM retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        "created_at": datetime.now().isoformat(),
        "documents": [],
        "total_chars": 0,
    }
    return new_session_id, sessions[new_session_id]


@app.get("/")
async def root():
    """Root endpoint - returns API info."""
    return {
        "name": "RLM - Recursive Language Model",
        "version": "1.0.0",
        "description": "Process arbitrarily long documents using recursive LLM retrieval",
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "sessions": "GET /sessions/{session_id}",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(sessions)}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload a document for RLM processing.
    
    Supports: PDF, TXT, DOCX, MD, JSON, and code files
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate or use session ID
        session_id, session = get_or_create_session(session_id)
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Save file
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        saved_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        async with aiofiles.open(saved_path, 'wb') as f:
            await f.write(content)
        
        # Process document
        document = await doc_processor.load_document(
            file_path=saved_path,
            content=content,
            filename=file.filename
        )
        
        # Store document info in session
        doc_info = {
            "id": file_id,
            "filename": file.filename,
            "doc_type": document.doc_type,
            "total_chars": document.total_chars,
            "num_chunks": document.num_chunks,
            "metadata": document.metadata,
            "saved_path": str(saved_path),
            "uploaded_at": datetime.now().isoformat(),
        }
        
        session["documents"].append(doc_info)
        session["total_chars"] += document.total_chars
        
        return {
            "success": True,
            "session_id": session_id,
            "document": doc_info,
            "message": f"Successfully uploaded and processed '{file.filename}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """Upload multiple documents at once."""
    session_id, session = get_or_create_session(session_id)
    results = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            
            if len(content) == 0:
                errors.append(f"{file.filename}: Empty file")
                continue
            
            # Save file
            file_id = str(uuid.uuid4())
            file_ext = Path(file.filename).suffix
            saved_path = UPLOAD_DIR / f"{file_id}{file_ext}"
            
            async with aiofiles.open(saved_path, 'wb') as f:
                await f.write(content)
            
            # Process document
            document = await doc_processor.load_document(
                file_path=saved_path,
                content=content,
                filename=file.filename
            )
            
            doc_info = {
                "id": file_id,
                "filename": file.filename,
                "doc_type": document.doc_type,
                "total_chars": document.total_chars,
                "num_chunks": document.num_chunks,
                "metadata": document.metadata,
                "saved_path": str(saved_path),
                "uploaded_at": datetime.now().isoformat(),
            }
            
            session["documents"].append(doc_info)
            session["total_chars"] += document.total_chars
            results.append(doc_info)
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    
    return {
        "success": True,
        "session_id": session_id,
        "documents": results,
        "errors": errors,
        "message": f"Successfully uploaded {len(results)} documents, {len(errors)} errors"
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        documents=session["documents"],
        total_chars=session["total_chars"]
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated files."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Delete uploaded files
    for doc in session["documents"]:
        try:
            path = Path(doc["saved_path"])
            if path.exists():
                path.unlink()
        except:
            pass
    
    del sessions[session_id]
    
    return {"success": True, "message": "Session deleted"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RLM retrieval.
    
    This is the core RLM endpoint that processes queries using
    recursive LLM calls for efficient long-context retrieval.
    """
    import time
    
    start_time = time.time()
    
    # Validate session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    if not session["documents"]:
        raise HTTPException(status_code=400, detail="No documents uploaded in this session")
    
    try:
        # Load documents from disk
        documents = []
        for doc_info in session["documents"]:
            doc = await doc_processor.load_document(doc_info["saved_path"])
            documents.append(doc)
        
        # Combine documents if multiple
        if len(documents) == 1:
            combined_doc = documents[0]
        else:
            combined_doc = doc_processor.combine_documents(documents)
        
        # Get context for RLM
        context = combined_doc.get_context_for_rlm()
        
        # Initialize LLM clients
        # Use environment variables for API keys
        root_provider = os.getenv("RLM_ROOT_PROVIDER", "mock")
        sub_provider = os.getenv("RLM_SUB_PROVIDER", "mock")
        root_model = os.getenv("RLM_ROOT_MODEL", "gpt-4o-mini")
        sub_model = os.getenv("RLM_SUB_MODEL", "gpt-4o-mini")
        
        root_client = LLMClientFactory.create(
            provider=root_provider,
            model=root_model
        )
        sub_client = LLMClientFactory.create(
            provider=sub_provider,
            model=sub_model
        )
        
        # Initialize RLM Engine
        if request.use_subcalls:
            rlm = RLMEngine(
                root_llm_client=root_client,
                sub_llm_client=sub_client,
                max_iterations=request.max_iterations,
                max_repl_output_chars=2000,
                sub_llm_max_chars=500000,
            )
        else:
            rlm = RLMEngineNoSubCalls(
                root_llm_client=root_client,
                sub_llm_client=sub_client,
                max_iterations=request.max_iterations,
                max_repl_output_chars=2000,
            )
        
        # Run RLM
        result = await rlm.run(
            query=request.query,
            context=context,
            context_type=combined_doc.doc_type
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            success=result["success"],
            answer=result.get("answer"),
            error=result.get("error"),
            iterations=result["iterations"],
            sub_lm_calls=result.get("sub_lm_calls", 0),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return QueryResponse(
            success=False,
            error=str(e),
            iterations=0,
            sub_lm_calls=0,
            processing_time_seconds=round(processing_time, 2)
        )


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface."""
    html_content = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_content.exists():
        return html_content.read_text()
    else:
        # Return a simple message if frontend not built
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLM - Recursive Language Model</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .info { background: #f0f0f0; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>RLM - Recursive Language Model</h1>
            <div class="info">
                <h2>API is running!</h2>
                <p>The RLM backend is operational. Use the API endpoints to upload documents and query them.</p>
                <p>Key endpoints:</p>
                <ul>
                    <li><code>POST /upload</code> - Upload a document</li>
                    <li><code>POST /query</code> - Query documents using RLM</li>
                    <li><code>GET /health</code> - Health check</li>
                </ul>
            </div>
        </body>
        </html>
        """


# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
