"""
Recursive Language Model (RLM) Engine
======================================

This module implements the core RLM scaffold as described in the paper:
"Recursive Language Models" by Zhang et al. (2026)

Key components:
- REPL environment that holds the prompt as a variable
- Symbolic recursion via code execution
- Programmatic sub-LM calls
- Iterative refinement through metadata feedback
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
        
        try:
            response = await self.llm_client.complete(prompt, **kwargs)
            logger.info(f"[Sub-LM Call #{call_id}] Response length: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"[Sub-LM Call #{call_id}] Error: {str(e)}")
            return f"Error in sub-LM call: {str(e)}"


class RLMEngine:
    """
    Recursive Language Model Engine.
    
    Implements the RLM scaffold that:
    1. Loads the prompt as a variable in a REPL environment
    2. Allows the LLM to write code to examine and decompose the prompt
    3. Supports recursive sub-LM calls via llm_query function
    4. Iterates until a final answer is produced
    """
    
    def __init__(
        self,
        root_llm_client: Any,
        sub_llm_client: Any,
        max_iterations: int = 50,
        max_output_tokens_per_iteration: int = 8000,
        max_repl_output_chars: int = 2000,
        sub_llm_max_chars: int = 500000,
    ):
        """
        Initialize the RLM Engine.
        
        Args:
            root_llm_client: LLM client for the root model (orchestrator)
            sub_llm_client: LLM client for sub-LM calls (can be same as root)
            max_iterations: Maximum number of RLM iterations
            max_output_tokens_per_iteration: Max tokens per root LLM call
            max_repl_output_chars: Max characters of REPL output to show
            sub_llm_max_chars: Max characters for sub-LM prompts
        """
        self.root_llm_client = root_llm_client
        self.sub_llm_client = sub_llm_client
        self.max_iterations = max_iterations
        self.max_output_tokens_per_iteration = max_output_tokens_per_iteration
        self.max_repl_output_chars = max_repl_output_chars
        self.sub_llm_max_chars = sub_llm_max_chars
        
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
        state.set_variable("Final", None)
        
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
    
    def _truncate_output(self, output: str, max_chars: int = None) -> str:
        """Truncate output to fit within context window constraints."""
        max_chars = max_chars or self.max_repl_output_chars
        if len(output) <= max_chars:
            return output
        
        half = max_chars // 2
        return output[:half] + f"\n...[truncated {len(output) - max_chars} chars]...\n" + output[-half:]
    
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
    
    def _extract_final_answer(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract final answer from text using FINAL() or FINAL_VAR() tags.
        
        Args:
            text: The text to parse
            
        Returns:
            Tuple of (answer_type, answer_content) or (None, None)
        """
        # Pattern for FINAL(answer)
        final_pattern = r'FINAL\((.*?)\)(?!\))'
        match = re.search(final_pattern, text, re.DOTALL)
        if match:
            return "direct", match.group(1).strip()
        
        # Pattern for FINAL_VAR(variable_name)
        final_var_pattern = r'FINAL_VAR\((\w+)\)'
        match = re.search(final_var_pattern, text)
        if match:
            return "variable", match.group(1).strip()
        
        return None, None
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from the LLM response."""
        # Match ```repl or ```python code blocks
        patterns = [
            r'```repl\s*\n(.*?)\n```',
            r'```python\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            codes.extend(matches)
        
        return codes
    
    async def run(
        self, 
        query: str, 
        context: Any,
        system_prompt: Optional[str] = None,
        context_type: str = "string"
    ) -> Dict[str, Any]:
        """
        Run the RLM scaffold on a query with the given context.
        
        Args:
            query: The user's query/question
            context: The context/prompt (can be very large)
            system_prompt: Optional custom system prompt
            context_type: Type description of the context
            
        Returns:
            Dictionary with results and metadata
        """
        # Initialize REPL environment
        env, state = self._create_repl_environment(context, context_type)
        self._inject_llm_query(env)
        
        # Get context metadata
        metadata = self._get_context_metadata(context)
        
        # Build conversation history
        history = []
        
        # Use provided system prompt or default
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(**metadata)
        
        # Initial user message
        initial_message = self._build_initial_message(query, metadata)
        history.append({"role": "user", "content": initial_message})
        
        logger.info(f"Starting RLM loop. Context: {metadata['context_total_length']} chars, "
                   f"{metadata['num_chunks']} chunks")
        
        # RLM Loop (Algorithm 1 from paper)
        for iteration in range(self.max_iterations):
            state.iteration = iteration
            logger.info(f"=== RLM Iteration {iteration + 1} ===")
            
            # Get response from root LLM
            try:
                response = await self.root_llm_client.complete(
                    messages=history,
                    system=system_prompt,
                    max_tokens=self.max_output_tokens_per_iteration
                )
            except Exception as e:
                logger.error(f"Root LLM error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Root LLM error: {str(e)}",
                    "iterations": iteration,
                    "history": history
                }
            
            # Check for final answer
            final_type, final_content = self._extract_final_answer(response)
            
            if final_type == "direct":
                logger.info(f"Final answer received (direct) at iteration {iteration + 1}")
                return {
                    "success": True,
                    "answer": final_content,
                    "iterations": iteration + 1,
                    "sub_lm_calls": self.sub_lm_invoker.call_count,
                    "history": history,
                    "final_type": "direct"
                }
            
            elif final_type == "variable":
                # Retrieve variable from REPL environment
                if final_content in env:
                    var_value = env[final_content]
                    # Convert to string if needed
                    if not isinstance(var_value, str):
                        var_value = str(var_value)
                    logger.info(f"Final answer received (variable: {final_content}) at iteration {iteration + 1}")
                    return {
                        "success": True,
                        "answer": var_value,
                        "iterations": iteration + 1,
                        "sub_lm_calls": self.sub_lm_invoker.call_count,
                        "history": history,
                        "final_type": "variable",
                        "variable_name": final_content
                    }
                else:
                    error_msg = f"Variable '{final_content}' not found in REPL environment"
                    logger.error(error_msg)
                    history.append({"role": "assistant", "content": response})
                    history.append({"role": "user", "content": f"Error: {error_msg}. Please check the variable name and try again."})
                    continue
            
            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response)
            
            if code_blocks:
                # Execute all code blocks
                all_outputs = []
                for code in code_blocks:
                    output, success = self._execute_code(code, env)
                    all_outputs.append(output)
                
                combined_output = "\n".join(all_outputs)
                truncated_output = self._truncate_output(combined_output)
                
                # Update history
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "user", 
                    "content": f"REPL Output:\n```\n{truncated_output}\n```"
                })
            else:
                # No code to execute, just continue conversation
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "user",
                    "content": "Please provide code to execute in the REPL environment using ```repl blocks, or provide a FINAL() answer when complete."
                })
        
        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached without final answer")
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "iterations": self.max_iterations,
            "sub_lm_calls": self.sub_lm_invoker.call_count,
            "history": history
        }
    
    def _get_default_system_prompt(
        self,
        context_type: str,
        context_total_length: int,
        context_lengths: List[int],
        num_chunks: int,
        **kwargs
    ) -> str:
        """
        Generate the default RLM system prompt as described in the paper.
        Based on Appendix C.1 from the paper.
        """
        prompt = f"""You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths[:10]}{'...' if len(context_lengths) > 10 else ''}.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context.
Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:

```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.

```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. <100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk:

```repl
query = "A man became famous for his book 'The Great Gatsby'. How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 5 chunks
chunk_size = len(context) // 10
answers = []
for i in range(10):
    if i < 9:
        chunk_str = "\\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\\n".join(context[i*chunk_size:])
    answer = llm_query(f"Try to answer the following query: {{query}}. Here are the documents:\\n{{chunk_str}}. Only answer if you are confident in your answer based on the evidence.")
    answers.append(answer)
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:

```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer."""
        
        return prompt
    
    def _build_initial_message(self, query: str, metadata: Dict[str, Any]) -> str:
        """Build the initial user message with the query."""
        return f"Query: {query}\n\nThe context has been loaded into the REPL environment as the 'context' variable. Begin by examining the context and developing a strategy to answer the query."


class RLMEngineNoSubCalls(RLMEngine):
    """
    RLM Engine without sub-LM calls (ablation from the paper).
    Used for comparison purposes.
    """
    
    def _inject_llm_query(self, env: Dict[str, Any]) -> None:
        """Override to not inject llm_query function."""
        pass
    
    def _get_default_system_prompt(self, **metadata) -> str:
        """Generate system prompt without sub-LM call instructions."""
        prompt = f"""You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a {metadata.get('context_type', 'string')} with {metadata.get('context_total_length', 0)} total characters, and is broken up into chunks of char lengths: {metadata.get('context_lengths', [])[:10]}{'...' if len(metadata.get('context_lengths', [])) > 10 else ''}.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment to not overflow the context window. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and save information to buffers.

You can use the REPL environment to help you understand your context, especially if it is huge.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want to peek at the first 10000 characters of the context:

```repl
chunk = context[:10000]
print(f"First 10000 characters of context: {{chunk}}")
```

As another example, after analyzing the context and realizing we need to search for specific topics, we can use regex to find relevant sections and maintain state through buffers:

```repl
# After finding out we need to search for "magic" and "number" in the context
import re
query_terms = ["magic", "number"]
relevant_sections = []
buffers = []
# Search for sections containing our query terms
for i, chunk in enumerate(context):
    chunk_text = str(chunk).lower()
    if any(term in chunk_text for term in query_terms):
        relevant_sections.append((i, chunk))
# Process each relevant section and print findings
for section_idx, section_content in relevant_sections:
    print(f"Found relevant section {{section_idx}} containing magic/number references:")
    print(f"Content: {{section_content[:500]}}...")  # Print first 500 chars
    buffers.append(f"Section {{section_idx}}: Contains magic/number references")
print(f"Total relevant sections found: {{len(relevant_sections)}}")
print("Summary of findings:")
for buffer in buffers:
    print(f"- {{buffer}}")
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Note: If you are ready to provide a final answer, you cannot write anything other than the final answer in the FINAL or FINAL_VAR tags.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. Remember to explicitly answer the original query in your final answer."""
        
        return prompt
