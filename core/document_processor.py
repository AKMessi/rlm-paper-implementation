"""
Document Processor for RLM - HIGH PERFORMANCE VERSION
======================================================

Ultra-fast document processing with:
- Async processing
- Parallel chunking
- Optimized PDF parsing
- Streaming file handling
"""

import io
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging
import time

import aiofiles
from pypdf import PdfReader

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a processed document."""
    content: Union[str, List[str], Dict[str, Any]]
    metadata: Dict[str, Any]
    source: str
    doc_type: str
    total_chars: int
    num_chunks: int = 1
    processing_time: float = 0.0
    
    def get_context_for_rlm(self) -> Any:
        """Get the content in a format suitable for RLM processing."""
        return self.content


class DocumentProcessor:
    """
    HIGH PERFORMANCE document processor.
    
    Optimizations:
    - Async file reading
    - Streaming PDF parsing
    - Parallel chunk processing
    - Memory-efficient chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 150000,  # Larger chunks = fewer API calls
        chunk_overlap: int = 500,
        respect_boundaries: bool = True,
        max_workers: int = 4,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_boundaries = respect_boundaries
        self.max_workers = max_workers
        
    async def load_document(
        self, 
        file_path: Union[str, Path],
        content: Optional[bytes] = None,
        filename: Optional[str] = None
    ) -> Document:
        """
        Load and process a document - HIGH PERFORMANCE VERSION.
        
        Args:
            file_path: Path to the file
            content: Optional raw bytes (if already loaded)
            filename: Original filename
            
        Returns:
            Processed Document object
        """
        start_time = time.time()
        file_path = Path(file_path)
        filename = filename or file_path.name
        extension = Path(filename).suffix.lower()
        
        logger.info(f"[FAST] Loading document: {filename}")
        
        # Read content if not provided
        if content is None:
            content = await self._fast_read(file_path)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"[FAST] File size: {file_size_mb:.2f} MB")
        
        # Parse based on file type
        parse_start = time.time()
        
        if extension == '.pdf':
            doc = await self._fast_parse_pdf(content, filename)
        elif extension in ['.docx', '.doc']:
            doc = await self._fast_parse_docx(content, filename)
        elif extension in ['.txt', '.md', '.markdown']:
            doc = await self._fast_parse_text(content, filename)
        elif extension in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.ts', '.tsx', '.jsx']:
            doc = await self._fast_parse_code(content, filename)
        elif extension in ['.json']:
            doc = await self._fast_parse_json(content, filename)
        else:
            doc = await self._fast_parse_text(content, filename)
        
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time
        doc.processing_time = total_time
        
        logger.info(f"[FAST] Parsed in {parse_time:.2f}s, Total: {total_time:.2f}s, "
                   f"Chunks: {doc.num_chunks}, Chars: {doc.total_chars:,}")
        
        return doc
    
    async def _fast_read(self, file_path: Path) -> bytes:
        """Fast async file reading."""
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def _fast_parse_pdf(self, content: bytes, filename: str) -> Document:
        """FAST PDF parsing with pypdf."""
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            
            # Extract text from all pages in parallel batches
            total_pages = len(reader.pages)
            logger.info(f"[FAST] PDF has {total_pages} pages")
            
            # Process pages in batches for speed
            batch_size = 10
            all_text_parts = []
            
            for i in range(0, total_pages, batch_size):
                batch_end = min(i + batch_size, total_pages)
                batch_tasks = []
                
                for page_num in range(i, batch_end):
                    batch_tasks.append(self._extract_page_text(reader.pages[page_num]))
                
                batch_results = await asyncio.gather(*batch_tasks)
                all_text_parts.extend(batch_results)
            
            total_text = "\n\n".join(filter(None, all_text_parts))
            
            # Fast chunking
            chunks = self._fast_chunk_text(total_text)
            
            metadata = {
                "filename": filename,
                "num_pages": total_pages,
                "file_type": "pdf",
                "parsed_at": time.time(),
            }
            
            return Document(
                content=chunks,
                metadata=metadata,
                source=filename,
                doc_type="pdf",
                total_chars=len(total_text),
                num_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"[FAST] PDF parse error: {str(e)}")
            raise
    
    async def _extract_page_text(self, page) -> str:
        """Extract text from a single PDF page."""
        try:
            return page.extract_text() or ""
        except:
            return ""
    
    async def _fast_parse_docx(self, content: bytes, filename: str) -> Document:
        """FAST DOCX parsing."""
        if not DOCX_AVAILABLE:
            return await self._fast_parse_text(content, filename)
        
        try:
            doc_file = io.BytesIO(content)
            doc = DocxDocument(doc_file)
            
            # Extract all paragraphs quickly
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            all_text = "\n\n".join(paragraphs)
            
            chunks = self._fast_chunk_text(all_text)
            
            metadata = {
                "filename": filename,
                "num_paragraphs": len(paragraphs),
                "file_type": "docx",
            }
            
            return Document(
                content=chunks,
                metadata=metadata,
                source=filename,
                doc_type="docx",
                total_chars=len(all_text),
                num_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"[FAST] DOCX parse error: {str(e)}")
            return await self._fast_parse_text(content, filename)
    
    async def _fast_parse_text(self, content: bytes, filename: str) -> Document:
        """FAST text parsing."""
        text = self._decode_bytes(content)
        
        is_markdown = filename.endswith(('.md', '.markdown'))
        
        if is_markdown and self.respect_boundaries:
            chunks = self._fast_chunk_markdown(text)
        else:
            chunks = self._fast_chunk_text(text)
        
        metadata = {
            "filename": filename,
            "file_type": "markdown" if is_markdown else "text",
        }
        
        return Document(
            content=chunks,
            metadata=metadata,
            source=filename,
            doc_type="text",
            total_chars=len(text),
            num_chunks=len(chunks)
        )
    
    async def _fast_parse_code(self, content: bytes, filename: str) -> Document:
        """FAST code parsing."""
        text = self._decode_bytes(content)
        chunks = self._fast_chunk_code(text, filename)
        
        metadata = {
            "filename": filename,
            "file_type": "code",
            "language": Path(filename).suffix[1:],
        }
        
        return Document(
            content=chunks,
            metadata=metadata,
            source=filename,
            doc_type="code",
            total_chars=len(text),
            num_chunks=len(chunks)
        )
    
    async def _fast_parse_json(self, content: bytes, filename: str) -> Document:
        """FAST JSON parsing."""
        import json
        
        text = self._decode_bytes(content)
        
        try:
            data = json.loads(text)
            formatted = json.dumps(data, indent=2)
            
            if isinstance(data, list) and len(data) > 100:
                # Large array - chunk by items
                chunks = self._chunk_json_array(data)
            else:
                chunks = self._fast_chunk_text(formatted)
            
            metadata = {
                "filename": filename,
                "file_type": "json",
            }
            
            return Document(
                content=chunks,
                metadata=metadata,
                source=filename,
                doc_type="json",
                total_chars=len(formatted),
                num_chunks=len(chunks)
            )
            
        except json.JSONDecodeError:
            return await self._fast_parse_text(content, filename)
    
    def _chunk_json_array(self, data: list) -> List[str]:
        """Chunk large JSON arrays efficiently."""
        import json
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in data:
            item_str = json.dumps(item, indent=2)
            item_size = len(item_str)
            
            if current_size + item_size > self.chunk_size and current_chunk:
                chunks.append("[\n" + ",\n".join(current_chunk) + "\n]")
                current_chunk = []
                current_size = 0
            
            current_chunk.append(item_str)
            current_size += item_size
        
        if current_chunk:
            chunks.append("[\n" + ",\n".join(current_chunk) + "\n]")
        
        return chunks if chunks else [json.dumps(data, indent=2)]
    
    def _decode_bytes(self, content: bytes, errors: str = 'replace') -> str:
        """Fast bytes to string decoding."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        return content.decode('utf-8', errors=errors)
    
    def _fast_chunk_text(self, text: str) -> List[str]:
        """
        ULTRA-FAST text chunking.
        Simple and fast - no complex boundary detection.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to find a newline to break at (cleaner chunks)
            if end < text_len:
                # Look for newline within 500 chars of the end
                search_start = max(end - 500, start)
                newline_pos = text.rfind('\n', search_start, end)
                if newline_pos > search_start:
                    end = newline_pos + 1
            
            chunks.append(text[start:end])
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < text_len else end
        
        return chunks
    
    def _fast_chunk_markdown(self, text: str) -> List[str]:
        """Fast markdown chunking by headers."""
        # Split on headers
        header_pattern = r'\n(#{1,6}\s+.+\n)'
        parts = re.split(header_pattern, text)
        
        if len(parts) <= 1:
            return self._fast_chunk_text(text)
        
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            if len(current_chunk) + len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _fast_chunk_code(self, text: str, filename: str) -> List[str]:
        """Fast code chunking."""
        # Simple approach: chunk by lines, respecting size
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep last few lines for context
                current_chunk = current_chunk[-5:] if len(current_chunk) > 5 else []
                current_size = sum(len(l) + 1 for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def combine_documents(self, documents: List[Document]) -> Document:
        """
        Combine multiple documents into a single document for RLM processing.
        HIGH PERFORMANCE VERSION.
        """
        all_chunks = []
        total_chars = 0
        sources = []
        
        for doc in documents:
            if isinstance(doc.content, list):
                for i, chunk in enumerate(doc.content):
                    prefixed = f"=== {doc.source} (part {i+1}) ===\n\n{chunk}"
                    all_chunks.append(prefixed)
            else:
                prefixed = f"=== {doc.source} ===\n\n{str(doc.content)}"
                all_chunks.append(prefixed)
            
            total_chars += doc.total_chars
            sources.append(doc.source)
        
        metadata = {
            "sources": sources,
            "num_documents": len(documents),
            "combined": True,
        }
        
        return Document(
            content=all_chunks,
            metadata=metadata,
            source="combined_documents",
            doc_type="combined",
            total_chars=total_chars,
            num_chunks=len(all_chunks)
        )
