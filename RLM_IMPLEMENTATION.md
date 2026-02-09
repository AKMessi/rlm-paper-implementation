# True RLM Implementation

This document describes the TRUE implementation of Recursive Language Models (RLM) as described in the paper "Recursive Language Models" by Zhang et al. (2026).

## Algorithm 1 from the Paper

```
Algorithm 1: A recursive language model, around LLM M

Input: prompt P
Output: response Y

1:  state ← InitREPL(prompt=P)
2:  state ← AddFunction(state, sub_RLM M)
3:  hist ← [Metadata(state)]
4:  while True do
5:    code ← LLM_M(hist)
6:    (state, stdout) ← REPL(state, code)
7:    hist ← hist ∥ code ∥ Metadata(stdout)
8:    if state[Final] is set then
9:      return state[Final]
```

## Key Design Principles

### 1. **Prompt is in the Environment, Not Context Window**
- The user prompt P is loaded as a variable in a REPL environment
- The root LLM is given only **metadata** about P (length, type, etc.)
- The LLM never sees the full prompt directly - it must access it programmatically

### 2. **LLM Generates Only Code**
- Unlike chat-based systems, the LLM generates **only Python code**
- No conversational text, no explanations outside code blocks
- The code manipulates the context variable to perform analysis

### 3. **Symbolic Recursion via Sub-LM Calls**
- The REPL environment includes `llm_query()` function
- LLM can call sub-LMs programmatically from within loops
- Enables O(|P|) or even O(|P|²) semantic work

### 4. **Constant-Size History**
- History contains: code + stdout metadata (truncated)
- History does NOT grow unboundedly with conversation
- Each iteration adds constant-size information

### 5. **Termination via Final Variable**
- Loop terminates when `Final` variable is set in REPL state
- No parsing of `FINAL()` text tags
- Clean programmatic termination

## Implementation Details

### RLMEngine Class

Located in `core/rlm_engine.py`:

```python
class RLMEngine:
    def __init__(
        self,
        root_llm_client,      # LLM for generating code
        sub_llm_client,       # LLM for sub-queries
        max_iterations=50,    # Safety limit
        max_output_tokens_per_iteration=8000,
        max_stdout_metadata_chars=500,  # Constant-size!
        sub_llm_max_chars=500000,
    )
```

### Main Loop

```python
async def run(self, query, context):
    # 1. Initialize REPL
    env, state = self._create_repl_environment(context)
    self._inject_llm_query(env)
    
    # 2. Build initial history with metadata only
    metadata = self._get_context_metadata(context)
    history = self._build_initial_history(metadata, query)
    
    # 3. RLM Loop (Algorithm 1)
    for iteration in range(self.max_iterations):
        # Line 5: LLM generates code based on history
        code = await self.root_llm_client.complete(
            prompt=f"{system_prompt}\n\n{history}\n\nGenerate Python code:"
        )
        
        # Line 6: Execute code in REPL
        (state, stdout) = self._execute_code(code, env)
        
        # Line 7: Update history (constant-size metadata!)
        stdout_metadata = self._format_stdout_metadata(stdout)
        history += f"\n\n[Iteration {iteration}]\n{code}\n\n{stdout_metadata}"
        
        # Lines 8-9: Check for termination
        if "Final" in env:
            return env["Final"]
```

## Comparison: Old vs True RLM

| Aspect | Old Implementation | True RLM (Paper) |
|--------|-------------------|------------------|
| **LLM Output** | Conversational text + code | **Only Python code** |
| **History** | Chat messages (grows unbounded) | **Text with constant-size metadata** |
| **Termination** | Parse `FINAL()` from text | **Check `Final` variable in REPL** |
| **API Style** | Chat completion | **Text completion** |
| **Context Access** | LLM sees prompt via messages | **LLM only sees metadata, must access via code** |
| **Rate of Growth** | O(iterations × conversation) | **O(iterations × constant)** |

## System Prompt

The system prompt instructs the LLM to generate only code:

```python
SYSTEM_PROMPT = """You are a Recursive Language Model (RLM). Your task is to answer a query by writing Python code...

YOUR TASK:
Write Python code to analyze the context and answer the query...

IMPORTANT RULES:
1. Generate ONLY executable Python code wrapped in ```python blocks
2. Do NOT generate conversational text or explanations
3. When you have completed your analysis, assign it to the variable `Final`
4. Example: `Final = "The answer is 42 based on the analysis..."`
5. Once `Final` is set, the system will return it as the answer
"""
```

## Usage Example

```python
from core.rlm_engine import RLMEngine
from core.llm_client import OpenAIClient

# Create clients
root_client = OpenAIClient(model="gpt-4o")
sub_client = OpenAIClient(model="gpt-4o-mini")

# Initialize RLM
rlm = RLMEngine(
    root_llm_client=root_client,
    sub_llm_client=sub_client,
    max_iterations=10,
)

# Run query
result = await rlm.run(
    query="What are the main topics in this document?",
    context="Very long document text..."  # Can be millions of characters
)

print(result["answer"])  # The Final variable value
print(result["iterations"])  # Number of RLM iterations
print(result["sub_lm_calls"])  # Number of sub-LM calls made
```

## Rate Limiting

The implementation includes several rate limiting mechanisms:

1. **Exponential Backoff**: Gemini client retries on 429 errors with exponential backoff
2. **Delay Between Iterations**: Configurable delay (default 0.5s) between root LLM calls
3. **Delay Between Sub-Calls**: Small delay (0.1s) between sub-LM calls
4. **Max Iterations**: Hard limit (default 50) to prevent infinite loops

## Ablation: RLMEngineNoSubCalls

For comparison purposes, the paper includes an ablation without sub-calls:

```python
rlm = RLMEngineNoSubCalls(
    root_llm_client=root_client,
    sub_llm_client=sub_client,
)
```

This version:
- Does not inject `llm_query()` into the REPL
- Forces the root LLM to process everything via code
- Still benefits from REPL state persistence

## Testing

Run tests:
```bash
cd rlm_app
python -m pytest tests/test_rlm_engine.py -v
```

## Key Benefits of True RLM

1. **Unbounded Input Size**: Can process prompts far exceeding context window
2. **Unbounded Output Size**: Final answer can be arbitrarily long (stored in variable)
3. **Unbounded Semantic Work**: Can perform O(|P|) or O(|P|²) operations via loops
4. **Constant Memory**: History stays compact due to constant-size metadata
5. **Programmatic Control**: LLM controls flow via code, not just text generation

## References

- Paper: "Recursive Language Models" by Zhang et al. (2026)
- Algorithm 1: Core RLM scaffold (Section 2)
- Appendix C: System prompts
- Appendix A: Training details
