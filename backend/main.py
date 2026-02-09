"""
RLM FastAPI Backend
===================

REST API for the Recursive Language Model application.
Supports:
- Document upload (PDF, TXT, DOCX, MD, code files)
- Querying with RLM retrieval
- Session management
- User-provided API keys (no cost to deployer)
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
    # User-provided API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    root_provider: str = "openai"
    sub_provider: str = "openai"
    root_model: str = "gpt-4o"
    sub_model: str = "gpt-4o-mini"

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

class APIKeyRequest(BaseModel):
    session_id: str
    # All supported API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None  # Moonshot AI
    groq_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    # Provider and model selection
    root_provider: str = "openai"
    sub_provider: str = "openai"
    root_model: str = "gpt-4o-mini"
    sub_model: str = "gpt-4o-mini"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("=" * 60)
    print("RLM Application Starting")
    print("User brings their own API keys - no cost to deployer!")
    print("=" * 60)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Chunk size: {doc_processor.chunk_size}")
    print("=" * 60)
    yield
    # Shutdown
    print("Shutting down RLM Application")


app = FastAPI(
    title="RLM - Recursive Language Model",
    description="Process arbitrarily long documents using recursive LLM retrieval. Users provide their own API keys.",
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
        "api_keys": {},  # Store user API keys per session
    }
    return new_session_id, sessions[new_session_id]


@app.get("/")
async def root():
    """Root endpoint - redirect to web UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(sessions), "byok": True}


@app.post("/api/keys")
async def set_api_keys(request: APIKeyRequest):
    """
    Store API keys for a session (in memory only).
    Keys are NOT stored permanently - they disappear when session expires.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    # Store all API keys in session (memory only)
    session["api_keys"] = {
        "openai_api_key": request.openai_api_key,
        "anthropic_api_key": request.anthropic_api_key,
        "groq_api_key": request.groq_api_key,
        "together_api_key": request.together_api_key,
        "google_api_key": request.google_api_key,
        "mistral_api_key": request.mistral_api_key,
        "cohere_api_key": request.cohere_api_key,
        "deepseek_api_key": request.deepseek_api_key,
        "azure_api_key": request.azure_api_key,
        "perplexity_api_key": request.perplexity_api_key,
        "kimi_api_key": request.kimi_api_key,
        "root_provider": request.root_provider,
        "sub_provider": request.sub_provider,
        "root_model": request.root_model,
        "sub_model": request.sub_model,
    }
    
    # Helper to mask key
    def mask_key(key):
        if key and len(key) > 12:
            return key[:8] + "..." + key[-4:]
        return None
    
    # Mask keys for response
    keys_set = {
        "openai": mask_key(request.openai_api_key),
        "anthropic": mask_key(request.anthropic_api_key),
        "groq": mask_key(request.groq_api_key),
        "together": mask_key(request.together_api_key),
        "google": mask_key(request.google_api_key),
        "mistral": mask_key(request.mistral_api_key),
        "cohere": mask_key(request.cohere_api_key),
        "deepseek": mask_key(request.deepseek_api_key),
        "azure": mask_key(request.azure_api_key),
        "perplexity": mask_key(request.perplexity_api_key),
        "kimi/moonshot": mask_key(request.kimi_api_key),
    }
    
    # Filter out None values
    keys_set = {k: v for k, v in keys_set.items() if v is not None}
    
    return {
        "success": True,
        "message": "API keys stored in session memory",
        "keys_set": keys_set,
        "providers": {
            "root": request.root_provider,
            "sub": request.sub_provider,
        },
        "models": {
            "root": request.root_model,
            "sub": request.sub_model,
        }
    }


@app.get("/api/keys/{session_id}")
async def check_api_keys(session_id: str):
    """Check if API keys are set for a session (returns masked keys only)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    api_keys = session.get("api_keys", {})
    
    openai_key = api_keys.get("openai_api_key")
    anthropic_key = api_keys.get("anthropic_api_key")
    
    return {
        "has_openai": openai_key is not None,
        "has_anthropic": anthropic_key is not None,
        "masked_openai": openai_key[:8] + "..." + openai_key[-4:] if openai_key else None,
        "masked_anthropic": anthropic_key[:8] + "..." + anthropic_key[-4:] if anthropic_key else None,
        "providers": {
            "root": api_keys.get("root_provider", "openai"),
            "sub": api_keys.get("sub_provider", "openai"),
        },
        "models": {
            "root": api_keys.get("root_model", "gpt-4o"),
            "sub": api_keys.get("sub_model", "gpt-4o-mini"),
        }
    }


@app.delete("/api/keys/{session_id}")
async def clear_api_keys(session_id: str):
    """Clear API keys from session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions[session_id]["api_keys"] = {}
    return {"success": True, "message": "API keys cleared from session"}


@app.get("/api/models")
async def get_available_models():
    """Get list of available models and providers."""
    from core.llm_client import LLMClientFactory
    
    provider_info = LLMClientFactory.get_provider_info()
    
    return {
        "providers": list(provider_info.keys()),
        "providers_detail": provider_info,
        "recommendations": {
            "cost_effective": "deepseek-chat or groq-llama-3.1-8b for root + sub",
            "best_quality": "gpt-4o or claude-3-5-sonnet for root + gpt-4o-mini for sub",
            "balanced": "groq-llama-3.1-70b for root + groq-llama-3.1-8b for sub",
            "free_local": "ollama with llama3.1 (completely free!)"
        },
        "cost_ranking": {
            "cheapest": ["mock", "ollama", "deepseek", "groq", "together"],
            "mid_range": ["google", "mistral", "cohere", "openai (4o-mini)"],
            "premium": ["openai (4o)", "anthropic", "azure"]
        }
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    FAST document upload for RLM processing.
    
    Supports: PDF, TXT, DOCX, MD, JSON, and code files
    """
    import time
    start_time = time.time()
    
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
        
        # Check max file size (100MB now - optimized for speed)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: 100MB")
        
        # Save file asynchronously
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        saved_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        # Fast async write
        async with aiofiles.open(saved_path, 'wb') as f:
            await f.write(content)
        
        # Process document with HIGH PERFORMANCE processor
        document = await doc_processor.load_document(
            file_path=saved_path,
            content=content,
            filename=file.filename
        )
        
        total_time = time.time() - start_time
        
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
            "processing_time_seconds": round(document.processing_time, 2),
            "upload_time_seconds": round(total_time, 2),
        }
        
        session["documents"].append(doc_info)
        session["total_chars"] += document.total_chars
        
        return {
            "success": True,
            "session_id": session_id,
            "document": doc_info,
            "message": f"✓ Processed '{file.filename}' in {total_time:.1f}s ({document.num_chunks} chunks, {document.total_chars:,} chars)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    FAST upload multiple documents at once.
    Optimized for speed with parallel processing.
    """
    import time
    start_time = time.time()
    
    session_id, session = get_or_create_session(session_id)
    results = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            file_size = len(content)
            
            if file_size == 0:
                errors.append(f"{file.filename}: Empty file")
                continue
            
            # 100MB limit for faster processing
            max_size = 100 * 1024 * 1024
            if file_size > max_size:
                errors.append(f"{file.filename}: File too large (max 100MB)")
                continue
            
            # Fast async save
            file_id = str(uuid.uuid4())
            file_ext = Path(file.filename).suffix
            saved_path = UPLOAD_DIR / f"{file_id}{file_ext}"
            
            async with aiofiles.open(saved_path, 'wb') as f:
                await f.write(content)
            
            # HIGH PERFORMANCE document processing
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
                "processing_time_seconds": round(document.processing_time, 2),
            }
            
            session["documents"].append(doc_info)
            session["total_chars"] += document.total_chars
            results.append(doc_info)
            
        except Exception as e:
            logger.error(f"Upload error for {file.filename}: {str(e)}")
            errors.append(f"{file.filename}: {str(e)}")
    
    total_time = time.time() - start_time
    
    return {
        "success": True,
        "session_id": session_id,
        "documents": results,
        "errors": errors,
        "message": f"Processed {len(results)} files in {total_time:.1f}s" if results else "Upload failed",
        "total_time_seconds": round(total_time, 2)
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
    
    Uses user-provided API keys from session, or falls back to environment variables.
    """
    import time
    
    start_time = time.time()
    
    # Validate session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    if not session["documents"]:
        raise HTTPException(status_code=400, detail="No documents uploaded in this session")
    
    # Get API keys - prioritize request body, then session, then environment
    api_keys = session.get("api_keys", {})
    
    root_provider = request.root_provider or api_keys.get("root_provider", "openai")
    sub_provider = request.sub_provider or api_keys.get("sub_provider", "openai")
    root_model = request.root_model or api_keys.get("root_model", "gpt-4o")
    sub_model = request.sub_model or api_keys.get("sub_model", "gpt-4o-mini")
    
    # Helper to get API key for a provider
    def get_key_for_provider(provider):
        provider = provider.lower()
        
        # Map provider names to their API key sources
        key_mapping = {
            "openai": (request.openai_api_key, "openai_api_key", "OPENAI_API_KEY"),
            "anthropic": (request.anthropic_api_key, "anthropic_api_key", "ANTHROPIC_API_KEY"),
            "google": (request.google_api_key, "google_api_key", "GOOGLE_API_KEY"),
            "gemini": (request.google_api_key, "google_api_key", "GOOGLE_API_KEY"),
            "kimi": (request.kimi_api_key, "kimi_api_key", "KIMI_API_KEY"),
            "moonshot": (request.kimi_api_key, "kimi_api_key", "MOONSHOT_API_KEY"),
            "groq": (request.groq_api_key, "groq_api_key", "GROQ_API_KEY"),
            "together": (request.together_api_key, "together_api_key", "TOGETHER_API_KEY"),
            "mistral": (request.mistral_api_key, "mistral_api_key", "MISTRAL_API_KEY"),
            "cohere": (request.cohere_api_key, "cohere_api_key", "COHERE_API_KEY"),
            "deepseek": (request.deepseek_api_key, "deepseek_api_key", "DEEPSEEK_API_KEY"),
            "perplexity": (request.perplexity_api_key, "perplexity_api_key", "PERPLEXITY_API_KEY"),
            "azure": (request.azure_api_key, "azure_api_key", "AZURE_OPENAI_KEY"),
        }
        
        if provider in key_mapping:
            req_key, session_key, env_key = key_mapping[provider]
            return req_key or api_keys.get(session_key) or os.getenv(env_key)
        elif provider == "ollama":
            return "ollama"
        elif provider == "mock":
            return "mock"
        
        return None
    
    root_key = get_key_for_provider(root_provider)
    sub_key = get_key_for_provider(sub_provider)
    
    # Validate API keys
    if root_provider not in ["ollama", "mock"] and not root_key:
        raise HTTPException(
            status_code=400, 
            detail=f"{root_provider.upper()} API key required. Please set your API key in the UI."
        )
    if sub_provider not in ["ollama", "mock"] and not sub_key:
        raise HTTPException(
            status_code=400,
            detail=f"{sub_provider.upper()} API key required for sub-model."
        )
    
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
        
        # Initialize LLM clients with user-provided keys
        root_client = LLMClientFactory.create(
            provider=root_provider,
            model=root_model,
            api_key=root_key
        )
        sub_client = LLMClientFactory.create(
            provider=sub_provider,
            model=sub_model,
            api_key=sub_key
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
        
    except HTTPException:
        raise
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
                .byok { background: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 20px; border-left: 4px solid #4caf50; }
            </style>
        </head>
        <body>
            <h1>RLM - Recursive Language Model</h1>
            <div class="info">
                <h2>API is running!</h2>
                <p>The RLM backend is operational.</p>
                <p>Key endpoints:</p>
                <ul>
                    <li><code>POST /upload</code> - Upload a document</li>
                    <li><code>POST /query</code> - Query documents using RLM</li>
                    <li><code>GET /health</code> - Health check</li>
                </ul>
            </div>
            <div class="byok">
                <h3>Bring Your Own Key (BYOK)</h3>
                <p>This app uses your own OpenAI or Anthropic API keys. 
                No costs are incurred by the deployer.</p>
                <p>Set your keys via: <code>POST /api/keys</code></p>
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
