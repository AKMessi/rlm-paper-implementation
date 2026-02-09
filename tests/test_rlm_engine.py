"""
Tests for RLM Engine
====================

Unit tests for the core RLM functionality.
Tests the TRUE implementation of Algorithm 1 from the paper.
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
        max_stdout_metadata_chars=500,
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
    
    def test_final_variable(self):
        """Test that Final variable works as termination condition."""
        state = REPLState()
        assert not state.has_variable("Final")
        state.set_variable("Final", "The answer is 42")
        assert state.has_variable("Final")
        assert state.get_variable("Final") == "The answer is 42"


class TestRLMEngine:
    """Test the RLM Engine - TRUE Algorithm 1 implementation."""
    
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
```python
print("hello")
```
Some text after
"""
        blocks = rlm_engine._extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]
    
    def test_extract_code_blocks_repl(self, rlm_engine):
        """Test code block extraction with repl tag."""
        text = """
```repl
chunk = context[:100]
print(chunk)
```
"""
        blocks = rlm_engine._extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'chunk = context[:100]' in blocks[0]
    
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
    
    def test_format_metadata(self, rlm_engine):
        """Test metadata formatting for history."""
        metadata = {
            "context_type": "string",
            "context_total_length": 1000,
            "num_chunks": 1,
            "context_lengths": [1000]
        }
        formatted = rlm_engine._format_metadata(metadata)
        assert "Type: string" in formatted
        assert "Total length: 1000" in formatted
    
    def test_format_stdout_metadata(self, rlm_engine):
        """Test stdout metadata formatting (constant-size)."""
        # Short output
        short = "Hello world"
        formatted = rlm_engine._format_stdout_metadata(short)
        assert "Hello world" in formatted
        
        # Long output (should be truncated)
        long_output = "x" * 2000
        formatted = rlm_engine._format_stdout_metadata(long_output)
        assert "truncated" in formatted
        assert len(formatted) < len(long_output) + 200  # Should be much shorter
    
    def test_repl_state_persistence(self, rlm_engine):
        """Test that REPL state persists across iterations."""
        context = "Test content"
        env, state = rlm_engine._create_repl_environment(context)
        
        # Execute code that sets a variable
        code = "my_var = 'test value'"
        output, success = rlm_engine._execute_code(code, env)
        
        assert success
        assert env["my_var"] == "test value"
        
        # Execute another code block that uses the variable
        code2 = "print(my_var)"
        output2, success2 = rlm_engine._execute_code(code2, env)
        
        assert success2
        assert "test value" in output2
    
    def test_final_variable_detection(self, rlm_engine):
        """Test that Final variable is correctly detected."""
        context = "Test content"
        env, state = rlm_engine._create_repl_environment(context)
        
        # Initially no Final variable in env
        assert "Final" not in env
        
        # Execute code that sets Final
        code = 'Final = "This is the answer"'
        output, success = rlm_engine._execute_code(code, env)
        
        assert success
        # Final is stored in env (REPL environment), not state
        assert "Final" in env
        assert env["Final"] == "This is the answer"


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
    
    @pytest.mark.asyncio
    async def test_ablation_runs(self, mock_root_client, mock_sub_client):
        """Test that ablation version can still run."""
        rlm = RLMEngineNoSubCalls(
            root_llm_client=mock_root_client,
            sub_llm_client=mock_sub_client,
            max_iterations=5,
        )
        
        result = await rlm.run("test query", "test context")
        
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
