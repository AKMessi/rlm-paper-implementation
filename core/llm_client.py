"""
LLM Client for RLM
==================

Unified interface for different LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
- Groq (Llama, Mixtral - fast & cheap)
- Google (Gemini)
- Cohere
- Mistral
- Together AI (open source models)
- DeepSeek
- Azure OpenAI
- Ollama (local models)
- Mock (for testing)
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.extra_params = kwargs
        
    @abstractmethod
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from the LLM."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=httpx.Timeout(120.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from OpenAI."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


class AzureOpenAIClient(BaseLLMClient):
    """Client for Azure OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_KEY"),
            base_url=f"{self.azure_endpoint}/openai/deployments/{model}",
            default_query={"api-version": api_version},
            timeout=httpx.Timeout(120.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Azure OpenAI."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            raise


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""
    
    def __init__(
        self, 
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Anthropic Claude."""
        
        if messages is None:
            messages = []
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
        anthropic_messages = []
        for msg in messages:
            if msg["role"] != "system":
                anthropic_messages.append(msg)
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **{**self.extra_params, **kwargs}
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


class GroqClient(BaseLLMClient):
    """
    Client for Groq API.
    Fast inference for open source models.
    Great for sub-LM calls (cheap and fast).
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Groq."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise


class TogetherClient(BaseLLMClient):
    """
    Client for Together AI API.
    Great for open source models like Llama, Mixtral, Qwen.
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1"
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Together AI."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Together AI API error: {str(e)}")
            raise


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Google Gemini."""
        
        # Convert messages to Gemini format
        contents = []
        if messages:
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        elif prompt:
            contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}:generateContent?key={self.api_key}",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


class MistralClient(BaseLLMClient):
    """Client for Mistral AI API."""
    
    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Mistral."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            raise


class CohereClient(BaseLLMClient):
    """Client for Cohere API."""
    
    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.base_url = "https://api.cohere.ai/v1"
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Cohere."""
        
        # Convert messages to Cohere format
        chat_history = []
        message = prompt or ""
        
        if messages:
            for i, msg in enumerate(messages[:-1] if len(messages) > 1 else []):
                if msg["role"] == "user":
                    chat_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg["content"]})
            
            if messages:
                last_msg = messages[-1]
                if last_msg["role"] == "user":
                    message = last_msg["content"]
        
        payload = {
            "model": self.model,
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_history": chat_history,
            **self.extra_params
        }
        
        if system:
            payload["preamble"] = system
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["text"]
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise


class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek API."""
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0)
        )
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from DeepSeek."""
        
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{**self.extra_params, **kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama (local models).
    Run models locally - completely free!
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model, api_key="ollama", **kwargs)
        self.base_url = base_url
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Get completion from Ollama."""
        
        # Build messages
        if messages is None:
            messages = []
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
        # Ollama format
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            # Insert system message at start
            payload["messages"].insert(0, {"role": "system", "content": system})
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # Longer timeout for local
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self, model: str = "mock-model", **kwargs):
        super().__init__(model, **kwargs)
        self.call_count = 0
        
    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Return a mock response."""
        self.call_count += 1
        
        prompt_text = prompt or ""
        if messages:
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    prompt_text += msg["content"]
        
        prompt_lower = prompt_text.lower()
        
        # Sub-LM calls
        if system is None and ("chunk" in prompt_lower or "section" in prompt_lower or "documents" in prompt_lower):
            return "I found relevant information in this chunk. The key points are: [simulated content analysis mentioning Ada Lovelace and Alan Turing]"
        
        elif "summarize" in prompt_lower:
            return "Summary: [simulated summary of the provided text]"
        
        # Root RLM iterations
        if self.call_count <= 2:
            return '''```repl
# Initial exploration of the context
print("Exploring context...")
print(f"Type: {type(context)}")
print(f"Length: {len(context) if hasattr(context, '__len__') else 'N/A'}")

if isinstance(context, list):
    print(f"Number of items: {len(context)}")
    if len(context) > 0:
        print(f"First item preview: {str(context[0])[:200]}")
        
if isinstance(context, list) and len(context) >= 50:
    section_50 = context[49]
    print(f"Section 50 preview: {section_50[:300]}")
```
'''
        else:
            return '''Based on my analysis of the document, I found the relevant information in section 50.

FINAL(Section 50 discusses The History of Computing, mentioning Ada Lovelace, Alan Turing, mechanical calculators, and the ENIAC computer.)'''


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    # Provider display names for UI
    PROVIDER_INFO = {
        "openai": {
            "name": "OpenAI",
            "description": "GPT-4, GPT-4o, GPT-4o-mini",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "key_url": "https://platform.openai.com/api-keys",
            "cost": "~$0.50-5 per 1M tokens"
        },
        "azure": {
            "name": "Azure OpenAI",
            "description": "Enterprise OpenAI",
            "models": ["gpt-4o", "gpt-4", "gpt-35-turbo"],
            "key_url": "https://portal.azure.com",
            "cost": "Enterprise pricing"
        },
        "anthropic": {
            "name": "Anthropic",
            "description": "Claude 3.5 Sonnet, Claude 3 Haiku",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            "key_url": "https://console.anthropic.com/",
            "cost": "~$0.50-15 per 1M tokens"
        },
        "groq": {
            "name": "Groq",
            "description": "Fast Llama & Mixtral inference",
            "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "key_url": "https://console.groq.com/keys",
            "cost": "~$0.20-0.60 per 1M tokens"
        },
        "together": {
            "name": "Together AI",
            "description": "Open source models (Llama, Qwen, etc.)",
            "models": ["meta-llama/Llama-3.1-70B-Instruct-Turbo", "meta-llama/Llama-3.1-8B-Instruct-Turbo"],
            "key_url": "https://api.together.xyz/",
            "cost": "~$0.20-0.90 per 1M tokens"
        },
        "google": {
            "name": "Google Gemini",
            "description": "Gemini 1.5 Flash & Pro",
            "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "key_url": "https://aistudio.google.com/app/apikey",
            "cost": "~$0.08-3.50 per 1M tokens"
        },
        "mistral": {
            "name": "Mistral AI",
            "description": "Mistral Large & Medium",
            "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
            "key_url": "https://console.mistral.ai/",
            "cost": "~$0.20-4 per 1M tokens"
        },
        "cohere": {
            "name": "Cohere",
            "description": "Command R & R+",
            "models": ["command-r-plus", "command-r", "command"],
            "key_url": "https://dashboard.cohere.com/api-keys",
            "cost": "~$0.50-2.50 per 1M tokens"
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "DeepSeek Chat",
            "models": ["deepseek-chat", "deepseek-coder"],
            "key_url": "https://platform.deepseek.com/",
            "cost": "~$0.07-0.30 per 1M tokens"
        },
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Run models locally - FREE",
            "models": ["llama3.1", "mistral", "qwen2.5"],
            "key_url": "https://ollama.com/",
            "cost": "FREE (runs on your machine)"
        },
        "mock": {
            "name": "Mock (Testing)",
            "description": "Fake responses for testing",
            "models": ["mock-model"],
            "key_url": "N/A",
            "cost": "FREE"
        }
    }
    
    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, str]]:
        """Get information about all providers for UI."""
        return LLMClientFactory.PROVIDER_INFO
    
    @staticmethod
    def create(
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """Create an LLM client based on provider."""
        provider = provider.lower()
        
        if provider == "openai":
            model = model or "gpt-4o-mini"
            return OpenAIClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "azure":
            model = model or "gpt-4o"
            return AzureOpenAIClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "anthropic":
            model = model or "claude-3-5-sonnet-20241022"
            return AnthropicClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "groq":
            model = model or "llama-3.1-70b-versatile"
            return GroqClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "together":
            model = model or "meta-llama/Llama-3.1-70B-Instruct-Turbo"
            return TogetherClient(model=model, api_key=api_key, **kwargs)
        
        elif provider in ["google", "gemini"]:
            model = model or "gemini-1.5-flash"
            return GeminiClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "mistral":
            model = model or "mistral-large-latest"
            return MistralClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "cohere":
            model = model or "command-r-plus"
            return CohereClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "deepseek":
            model = model or "deepseek-chat"
            return DeepSeekClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "ollama":
            model = model or "llama3.1"
            return OllamaClient(model=model, **kwargs)
        
        elif provider == "mock":
            return MockLLMClient(model=model or "mock-model", **kwargs)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
