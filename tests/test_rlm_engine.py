"""
Tests for RLM Engine
====================

Unit tests for the core RLM functionality.
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from core.rlm_engine import RLMEngine, RLMEngineNoSubCalls, REPLState
from core.llm_client import MockLLMClient

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest.fixture
def mock_root_client():
    """Create a mock root LLM client."""
    return MockLLMClient(model="mock-root")


@pytest.fixture
def mock_sub_client():
    """Create a mock sub LLM client."""
    return MockLLMClient(model="mock-sub")


@pytest.fixture
def rlm_engine(mock_root_client, mock_sub_client):
    """Create an RLM engine with mock clients."""
    return RLMEngine(
        root_llm_client=mock_root_client,
        sub_llm_client=mock_sub_client,
        max_iterations=10,
        max_repl_output_chars=1000,
        sub_llm_max_chars=10000,
    )


class TestREPLState:
    """Test the REPLState class."""
    
    def test_initialization(self):
        state = REPLState()
        assert state.variables == {}
        assert state.stdout_history == []
        assert state.iteration == 0
        assert state.final_output is None
    
    def test_set_get_variable(self):
        state = REPLState()
        state.set_variable("test", "value")
        assert state.get_variable("test") == "value"
        assert state.has_variable("test")
        assert not state.has_variable("nonexistent")


class TestRLMEngine:
    """Test the RLM Engine."""
    
    @pytest.mark.asyncio
    async def test_simple_query(self, rlm_engine):
        """Test a simple query execution."""
        context = "This is a test context with some information."
        query = "What is in this context?"
        
        result = await rlm_engine.run(query, context)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "iterations" in result
        assert "history" in result
    
    @pytest.mark.asyncio
    async def test_chunked_context(self, rlm_engine):
        """Test with a context that requires chunking."""
        # Create a large context
        context = ["Chunk " + str(i) + ": " + "x" * 1000 for i in range(10)]
        query = "What are the main chunks?"
        
        result = await rlm_engine.run(query, context, context_type="list")
        
        assert isinstance(result, dict)
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_code_execution(self, rlm_engine):
        """Test that code execution works in the REPL."""
        context = "Test content"
        
        # The mock client returns code in its response
        result = await rlm_engine.run("test", context)
        
        # Should have executed at least one iteration
        assert result["iterations"] > 0
    
    def test_extract_code_blocks(self, rlm_engine):
        """Test code block extraction."""
        text = """
Some text before
```repl
print("hello")
```
Some text after
"""
        blocks = rlm_engine._extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]
    
    def test_extract_final_answer_direct(self, rlm_engine):
        """Test extracting direct final answer."""
        text = "Some reasoning...\n\nFINAL(This is the answer)"
        final_type, final_content = rlm_engine._extract_final_answer(text)
        
        assert final_type == "direct"
        assert final_content == "This is the answer"
    
    def test_extract_final_answer_variable(self, rlm_engine):
        """Test extracting variable final answer."""
        text = "Some reasoning...\n\nFINAL_VAR(my_variable)"
        final_type, final_content = rlm_engine._extract_final_answer(text)
        
        assert final_type == "variable"
        assert final_content == "my_variable"
    
    def test_get_context_metadata_string(self, rlm_engine):
        """Test metadata extraction for string context."""
        context = "Hello world"
        metadata = rlm_engine._get_context_metadata(context)
        
        assert metadata["context_type"] == "string"
        assert metadata["context_total_length"] == 11
        assert metadata["num_chunks"] == 1
    
    def test_get_context_metadata_list(self, rlm_engine):
        """Test metadata extraction for list context."""
        context = ["chunk1", "chunk2", "chunk3"]
        metadata = rlm_engine._get_context_metadata(context)
        
        assert metadata["context_type"] == "list"
        assert metadata["num_chunks"] == 3
    
    def test_truncate_output(self, rlm_engine):
        """Test output truncation."""
        long_output = "x" * 5000
        truncated = rlm_engine._truncate_output(long_output, max_chars=100)
        
        assert len(truncated) < len(long_output)
        assert "truncated" in truncated


class TestRLMEngineNoSubCalls:
    """Test the RLM Engine without sub-calls (ablation)."""
    
    @pytest.mark.asyncio
    async def test_no_subcalls(self, mock_root_client, mock_sub_client):
        """Test that sub-calls are not injected."""
        rlm = RLMEngineNoSubCalls(
            root_llm_client=mock_root_client,
            sub_llm_client=mock_sub_client,
            max_iterations=5,
        )
        
        env, state = rlm._create_repl_environment("test")
        rlm._inject_llm_query(env)
        
        # llm_query should not be in env
        assert "llm_query" not in env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
