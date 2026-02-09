"""
Recursive Language Model (RLM) Engine
======================================

This module implements the core RLM scaffold as described in the paper:
"Recursive Language Models" by Zhang et al. (2026)

Algorithm 1 from the paper:
1. state ← InitREPL(prompt=P)
2. state ← AddFunction(state, sub_RLM M)
3. hist ← [Metadata(state)]
4. while True:
5.   code ← LLM_M(hist)
6.   (state, stdout) ← REPL(state, code)
7.   hist ← hist ∥ code ∥ Metadata(stdout)
8.   if state[Final] is set:
9.     return state[Final]

Key components:
- REPL environment that holds the prompt as a variable
- Symbolic recursion via code execution
- Programmatic sub-LM calls
- Iterative refinement through metadata feedback
- Completion-style API (not chat), LLM generates only code
"""

import re
import io
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class REPLState:
    """Represents the state of the REPL environment."""
    variables: Dict[str, Any] = field(default_factory=dict)
    stdout_history: List[str] = field(default_factory=list)
    iteration: int = 0
    final_output: Optional[str] = None
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the REPL environment."""
        self.variables[name] = value
        
    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL environment."""
        return self.variables.get(name)
    
    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in the REPL environment."""
        return name in self.variables


class SubLMInvoker:
    """
    Handles recursive sub-LM calls within the REPL environment.
    This is injected into the REPL as the `llm_query` function.
    """
    
    def __init__(self, llm_client: Any, max_chars: int = 500000):
        self.llm_client = llm_client
        self.max_chars = max_chars
        self.call_count = 0
        self.delay_between_calls = 0.1  # Small delay to avoid rate limits
        
    async def __call__(self, prompt: str, **kwargs) -> str:
        """
        Execute a sub-LM call with the given prompt.
        
        Args:
            prompt: The prompt to send to the sub-LM
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            The response from the sub-LM
        """
        self.call_count += 1
        call_id = self.call_count
        
        logger.info(f"[Sub-LM Call #{call_id}] Prompt length: {len(prompt)} chars")
        
        # Truncate if exceeds max_chars
        if len(prompt) > self.max_chars:
            logger.warning(f"[Sub-LM Call #{call_id}] Truncating prompt from {len(prompt)} to {self.max_chars} chars")
            prompt = prompt[:self.max_chars] + "\n...[truncated]"
        
        # Rate limiting delay
        if self.delay_between_calls > 0:
            await asyncio.sleep(self.delay_between_calls)
        
        try:
            response = await self.llm_client.complete(prompt=prompt, **kwargs)
            logger.info(f"[Sub-LM Call #{call_id}] Response length: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"[Sub-LM Call #{call_id}] Error: {str(e)}")
            return f"Error in sub-LM call: {str(e)}"


class RLMEngine:
    """
    Recursive Language Model Engine - TRUE Implementation of Algorithm 1.
    
    Implements the RLM scaffold that:
    1. Loads the prompt as a variable in a REPL environment
    2. LLM generates ONLY code (not conversational text)
    3. Supports recursive sub-LM calls via llm_query function
    4. Iterates until Final variable is set in REPL state
    5. History contains: code + stdout metadata (compact)
    """
    
    def __init__(
        self,
        root_llm_client: Any,
        sub_llm_client: Any,
        max_iterations: int = 50,
        max_output_tokens_per_iteration: int = 8000,
        max_stdout_metadata_chars: int = 500,  # Paper: constant-size metadata
        sub_llm_max_chars: int = 500000,
        min_delay_between_calls: float = 0.5,
    ):
        """
        Initialize the RLM Engine.
        
        Args:
            root_llm_client: LLM client for the root model (orchestrator)
            sub_llm_client: LLM client for sub-LM calls (can be same as root)
            max_iterations: Maximum number of RLM iterations
            max_output_tokens_per_iteration: Max tokens per root LLM call
            max_stdout_metadata_chars: Max chars of stdout metadata to include
            sub_llm_max_chars: Max characters for sub-LM prompts
            min_delay_between_calls: Minimum delay between API calls
        """
        self.root_llm_client = root_llm_client
        self.sub_llm_client = sub_llm_client
        self.max_iterations = max_iterations
        self.max_output_tokens_per_iteration = max_output_tokens_per_iteration
        self.max_stdout_metadata_chars = max_stdout_metadata_chars
        self.sub_llm_max_chars = sub_llm_max_chars
        self.min_delay_between_calls = min_delay_between_calls
        
        # Create sub-LM invoker
        self.sub_lm_invoker = SubLMInvoker(sub_llm_client, max_chars=sub_llm_max_chars)
        
    def _create_repl_environment(
        self, 
        context: Any, 
        context_type: str = "string"
    ) -> Tuple[Dict[str, Any], REPLState]:
        """
        Create the initial REPL environment with the context variable.
        
        Args:
            context: The prompt/context to load as a variable
            context_type: Type description of the context
            
        Returns:
            Tuple of (environment_dict, REPLState)
        """
        state = REPLState()
        state.set_variable("context", context)
        # Note: We don't pre-set Final - it's set by the LLM when done
        
        # Create environment with safe builtins
        env = {
            "__builtins__": {
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "sorted": sorted,
                "reversed": reversed,
                "any": any,
                "all": all,
                "print": print,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "slice": slice,
                "type": type,
                "dir": dir,
                "vars": vars,
                "locals": locals,
                "globals": globals,
                "repr": repr,
                "help": help,
                "open": open,
            },
            "context": context,
            "state": state,
        }
        
        return env, state
    
    def _inject_llm_query(self, env: Dict[str, Any]) -> None:
        """Inject the llm_query function into the REPL environment."""
        # Create a synchronous wrapper for the async sub-LM invoker
        def llm_query_sync(prompt: str, **kwargs) -> str:
            try:
                # Try to get running loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, we need to use run_coroutine_threadsafe
                # or create a new thread to run the coroutine
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.sub_lm_invoker(prompt, **kwargs)
                    )
                    return future.result(timeout=120)
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(self.sub_lm_invoker(prompt, **kwargs))
        
        env["llm_query"] = llm_query_sync
    
    def _get_context_metadata(self, context: Any) -> Dict[str, Any]:
        """
        Extract metadata about the context for the system prompt.
        This is the "Metadata(state)" from Algorithm 1.
        
        Args:
            context: The context variable
            
        Returns:
            Dictionary with metadata
        """
        if isinstance(context, str):
            return {
                "context_type": "string",
                "context_total_length": len(context),
                "context_lengths": [len(context)],
                "num_chunks": 1,
            }
        elif isinstance(context, list):
            lengths = [len(str(item)) for item in context]
            total = sum(lengths)
            return {
                "context_type": "list",
                "context_total_length": total,
                "context_lengths": lengths,
                "num_chunks": len(context),
            }
        elif isinstance(context, dict):
            lengths = [len(str(v)) for v in context.values()]
            total = sum(lengths)
            return {
                "context_type": "dict",
                "context_total_length": total,
                "context_lengths": lengths,
                "num_chunks": len(context),
            }
        else:
            str_context = str(context)
            return {
                "context_type": type(context).__name__,
                "context_total_length": len(str_context),
                "context_lengths": [len(str_context)],
                "num_chunks": 1,
            }
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as a string for the history."""
        return f"""[Context Metadata]
Type: {metadata.get('context_type', 'unknown')}
Total length: {metadata.get('context_total_length', 0)} chars
Number of chunks: {metadata.get('num_chunks', 1)}
Chunk lengths: {metadata.get('context_lengths', [])[:5]}{'...' if len(metadata.get('context_lengths', [])) > 5 else ''}
[/Context Metadata]"""
    
    def _format_stdout_metadata(self, stdout: str) -> str:
        """
        Create constant-size metadata about stdout for history.
        This is key to Algorithm 1 - we don't include full stdout.
        """
        if not stdout:
            return "[stdout: empty]"
        
        length = len(stdout)
        if length <= self.max_stdout_metadata_chars:
            return f"[stdout: {length} chars]\n{stdout}\n[/stdout]"
        else:
            half = self.max_stdout_metadata_chars // 2
            return f"[stdout: {length} chars total, showing {self.max_stdout_metadata_chars}]\n{stdout[:half]}\n...[truncated {length - self.max_stdout_metadata_chars} chars]...\n{stdout[-half:]}\n[/stdout]"
    
    def _execute_code(self, code: str, env: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Execute Python code in the REPL environment.
        
        Args:
            code: Python code to execute
            env: The REPL environment
            
        Returns:
            Tuple of (stdout_output, success)
        """
        stdout_buffer = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
                # Execute the code
                exec(code, env)
            
            output = stdout_buffer.getvalue()
            return output, True
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, False
    
    def _build_system_prompt(self, metadata: Dict[str, Any], query: str) -> str:
        """
        Build the system prompt for the root LLM.
        The LLM is instructed to generate ONLY code, not conversational text.
        """
        return f"""You are a Recursive Language Model (RLM). Your task is to answer a query by writing Python code that manipulates a large context variable.

CONTEXT INFORMATION:
- Type: {metadata['context_type']}
- Total length: {metadata['context_total_length']} characters
- Number of chunks: {metadata['num_chunks']}
- Chunk lengths: {metadata['context_lengths'][:10]}{'...' if len(metadata['context_lengths']) > 10 else ''}

AVAILABLE IN THE REPL ENVIRONMENT:
1. `context` - The main context variable containing the data you need to analyze
2. `llm_query(prompt)` - A function to query a sub-LM (can handle ~500K chars)
3. `print()` - To output information
4. Any variables you create will persist across iterations

USER QUERY: {query}

YOUR TASK:
Write Python code to analyze the context and answer the query. You can:
- Examine the context (e.g., `print(len(context))`, `print(context[:1000])`)
- Break it into chunks and process iteratively
- Use `llm_query()` to delegate analysis of chunks to sub-LMs
- Store intermediate results in variables
- Build up your final answer

IMPORTANT RULES:
1. Generate ONLY executable Python code wrapped in ```python blocks
2. Do NOT generate conversational text or explanations
3. When you have completed your analysis and have a final answer, assign it to the variable `Final`
4. Example: `Final = "The answer is 42 based on the analysis..."`
5. Once `Final` is set, the system will return it as the answer
6. You can iterate multiple times - the REPL state persists
7. Use print() to see outputs (they will be shown to you in the next iteration)

EXAMPLE WORKFLOW:
```python
# First, explore the context
print(f"Context type: {{type(context)}}")
print(f"Context length: {{len(context)}}")
```

```python
# Then, chunk and analyze
chunk_size = 100000
answers = []
for i in range(0, len(context), chunk_size):
    chunk = context[i:i+chunk_size]
    answer = llm_query(f"Analyze this chunk: {{chunk[:5000]}}...")
    answers.append(answer)
    print(f"Chunk {{i//chunk_size}} analyzed")
```

```python
# Finally, aggregate and set Final
Final = llm_query(f"Based on these analyses: {{answers}}, answer the original query.")
```

Remember: Only output code in ```python blocks. Do not write explanations outside code blocks."""

    def _build_initial_history(self, metadata: Dict[str, Any], query: str) -> str:
        """Build the initial history string with metadata."""
        return f"""Query: {query}

{self._format_metadata(metadata)}

Begin by exploring the context and developing a strategy."""

    async def run(
        self, 
        query: str, 
        context: Any,
        system_prompt: Optional[str] = None,
        context_type: str = "string"
    ) -> Dict[str, Any]:
        """
        Run the RLM scaffold on a query with the given context.
        Implements Algorithm 1 from the paper exactly.
        
        Args:
            query: The user's query/question
            context: The context/prompt (can be very large)
            system_prompt: Optional custom system prompt
            context_type: Type description of the context
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        # Initialize REPL environment (Algorithm 1, line 1)
        env, state = self._create_repl_environment(context, context_type)
        self._inject_llm_query(env)  # Algorithm 1, line 2
        
        # Get context metadata
        metadata = self._get_context_metadata(context)
        
        # Build system prompt and initial history (Algorithm 1, line 3)
        if system_prompt is None:
            system_prompt = self._build_system_prompt(metadata, query)
        
        # History is a text string, not chat messages (key difference from chat API)
        history = self._build_initial_history(metadata, query)
        
        logger.info(f"Starting RLM loop (Algorithm 1). Context: {metadata['context_total_length']} chars, {metadata['num_chunks']} chunks")
        
        # RLM Loop (Algorithm 1, lines 4-9)
        for iteration in range(self.max_iterations):
            state.iteration = iteration
            logger.info(f"=== RLM Iteration {iteration + 1} ===")
            
            # Rate limiting
            if iteration > 0 and self.min_delay_between_calls > 0:
                await asyncio.sleep(self.min_delay_between_calls)
            
            # Algorithm 1, line 5: code ← LLM_M(hist)
            # LLM generates ONLY code based on history (completion-style)
            try:
                full_prompt = f"{system_prompt}\n\n{history}\n\nGenerate Python code to continue:"
                
                response = await self.root_llm_client.complete(
                    prompt=full_prompt,
                    max_tokens=self.max_output_tokens_per_iteration,
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"Root LLM error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Root LLM error: {str(e)}",
                    "iterations": iteration,
                    "history": history,
                    "sub_lm_calls": self.sub_lm_invoker.call_count,
                }
            
            # Extract code from response
            code_blocks = self._extract_code_blocks(response)
            
            if not code_blocks:
                # LLM didn't generate code - this is an error in our setup
                logger.warning(f"No code blocks found in iteration {iteration + 1}")
                # Add to history and continue, hoping it will generate code next time
                history += f"\n\n[Iteration {iteration + 1}]\n{response}\n\n[No executable code found. Please generate Python code in ```python blocks.]"
                continue
            
            # Execute all code blocks (Algorithm 1, line 6)
            all_outputs = []
            all_code = []
            for code in code_blocks:
                output, success = self._execute_code(code, env)
                all_outputs.append(output)
                all_code.append(code)
            
            combined_output = "\n".join(all_outputs)
            combined_code = "\n\n".join(all_code)
            
            # Algorithm 1, line 7: hist ← hist ∥ code ∥ Metadata(stdout)
            # Update history with code and stdout metadata (constant-size)
            stdout_metadata = self._format_stdout_metadata(combined_output)
            history += f"\n\n[Iteration {iteration + 1}]\n```python\n{combined_code}\n```\n\n{stdout_metadata}"
            
            # Algorithm 1, lines 8-9: if state[Final] is set: return state[Final]
            if state.has_variable("Final"):
                final_value = state.get_variable("Final")
                # Convert to string if needed
                if not isinstance(final_value, str):
                    final_value = str(final_value)
                
                processing_time = time.time() - start_time
                logger.info(f"Final answer received at iteration {iteration + 1}")
                logger.info(f"Processing time: {processing_time:.2f}s")
                
                return {
                    "success": True,
                    "answer": final_value,
                    "iterations": iteration + 1,
                    "sub_lm_calls": self.sub_lm_invoker.call_count,
                    "history": history,
                    "processing_time_seconds": processing_time,
                }
        
        # Max iterations reached without Final being set
        logger.warning(f"Max iterations ({self.max_iterations}) reached without Final being set")
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached without Final being set. "
                   f"The LLM did not assign a value to the 'Final' variable.",
            "iterations": self.max_iterations,
            "sub_lm_calls": self.sub_lm_invoker.call_count,
            "history": history,
            "processing_time_seconds": time.time() - start_time,
        }
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from the LLM response."""
        # Match ```python or ```repl code blocks
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```repl\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            codes.extend(matches)
        
        return codes


class RLMEngineNoSubCalls(RLMEngine):
    """
    RLM Engine without sub-LM calls (ablation from the paper).
    Used for comparison purposes.
    """
    
    def _inject_llm_query(self, env: Dict[str, Any]) -> None:
        """Override to not inject llm_query function."""
        pass
    
    def _build_system_prompt(self, metadata: Dict[str, Any], query: str) -> str:
        """Generate system prompt without sub-LM call instructions."""
        return f"""You are a Recursive Language Model (RLM) without sub-calls. Your task is to answer a query by writing Python code that manipulates a large context variable.

CONTEXT INFORMATION:
- Type: {metadata['context_type']}
- Total length: {metadata['context_total_length']} characters
- Number of chunks: {metadata['num_chunks']}

AVAILABLE IN THE REPL ENVIRONMENT:
1. `context` - The main context variable containing the data you need to analyze
2. `print()` - To output information
3. Any variables you create will persist across iterations

USER QUERY: {query}

YOUR TASK:
Write Python code to analyze the context and answer the query using only the tools available.

IMPORTANT RULES:
1. Generate ONLY executable Python code wrapped in ```python blocks
2. Do NOT generate conversational text or explanations
3. When you have completed your analysis and have a final answer, assign it to the variable `Final`
4. Example: `Final = "The answer is 42 based on the analysis..."`
5. Once `Final` is set, the system will return it as the answer

Remember: Only output code in ```python blocks. Do not write explanations outside code blocks."""
