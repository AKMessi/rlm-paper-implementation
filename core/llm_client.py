"""
LLM Client for RLM - ULTIMATE VERSION
======================================

Unified interface for 15+ LLM providers with ALL latest 2025 models:
- OpenAI (GPT-5, GPT-4.5, o3, o4-mini, o3-pro, o1)
- Anthropic (Claude Opus 4.6, Sonnet 4.5, Haiku 4.5, Claude 3.5)
- Google (Gemini 2.5/3.0 Pro/Flash, 1.5 Pro/Flash)
- Moonshot AI (Kimi K2.5, K3, k1.5)
- Groq (Llama 4, 3.1, Mixtral)
- Together AI (Llama 4, Qwen 3, DeepSeek V3)
- And more...

All models updated as of February 2026
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
    """Client for OpenAI API - Includes ALL latest 2025 models."""
    
    # Latest OpenAI models as of February 2026
    AVAILABLE_MODELS = [
        # GPT-5 Series (Latest)
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # GPT-4.5 Series
        "gpt-4.5-preview",
        "gpt-4.5",
        # o3/o4 Reasoning Series (Latest)
        "o3",
        "o3-mini",
        "o3-pro",
        "o4-mini",
        "o4-mini-high",
        "o1",
        "o1-mini",
        "o1-pro",
        # GPT-4o Series
        "gpt-4o",
        "gpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-latest",
        # Legacy
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    
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


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API - ALL latest 2025 models."""
    
    # Latest Anthropic models as of February 2026
    AVAILABLE_MODELS = [
        # Claude 4.6 Series (LATEST)
        "claude-opus-4-6-20251101",
        "claude-opus-4-6",
        # Claude 4.5 Series
        "claude-opus-4-5-20251101",
        "claude-opus-4-5",
        "claude-sonnet-4-5-20251101",
        "claude-sonnet-4-5",
        "claude-haiku-4-5-20251101",
        "claude-haiku-4-5",
        # Claude 3.5 Series (Current Gen)
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        # Claude 3 Series
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
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


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API - ALL latest 2025 models."""
    
    # Latest Gemini models as of February 2026
    AVAILABLE_MODELS = [
        # Gemini 3.0 Series (LATEST)
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        # Gemini 2.5 Series
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview",
        # Gemini 2.0 Series
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.0-flash-exp",
        # Gemini 1.5 Series
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        # Legacy
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
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


class KimiClient(BaseLLMClient):
    """
    Client for Moonshot AI (Kimi) API.
    Supports Kimi K2.5, K3, k1.5 and all latest models.
    """
    
    # Latest Kimi models as of February 2026
    AVAILABLE_MODELS = [
        # Kimi K3 Series (LATEST)
        "kimi-k3",
        "kimi-k3-0325",
        # Kimi K2.5 Series
        "kimi-k2.5",
        "kimi-k2.5-0120",
        "kimi-k2.5-long",
        # Kimi k1.5 Series (Reasoning)
        "kimi-k1.5",
        "kimi-k1.5-long",
        "kimi-k1.5-short",
        # Legacy
        "kimi-latest",
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ]
    
    def __init__(
        self,
        model: str = "kimi-k2.5",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.base_url = "https://api.moonshot.cn/v1"
        
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
        """Get completion from Kimi."""
        
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
            logger.error(f"Kimi API error: {str(e)}")
            raise


class GroqClient(BaseLLMClient):
    """
    Client for Groq API.
    Fast inference for open source models.
    """
    
    # Latest Groq models as of February 2026
    AVAILABLE_MODELS = [
        # Llama 4 Series (LATEST)
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        # Llama 3.3/3.1 Series
        "llama-3.3-70b-versatile",
        "llama-3.1-405b-reasoning",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        # Legacy
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma-2-9b-it",
        "gemma-7b-it",
    ]
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
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
    Great for open source models.
    """
    
    # Latest Together AI models as of February 2026
    AVAILABLE_MODELS = [
        # Llama 4 Series
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # Llama 3.3 Series
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        # DeepSeek
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-Coder-V2",
        # Qwen
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        # Mistral
        "mistralai/Mistral-Large-2",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        # Other
        "google/gemma-2-27b-it",
        "databricks/dbrx-instruct",
    ]
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
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


class MistralClient(BaseLLMClient):
    """Client for Mistral AI API."""
    
    AVAILABLE_MODELS = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "mistral-embed",
    ]
    
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


class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek API."""
    
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ]
    
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


class CohereClient(BaseLLMClient):
    """Client for Cohere API."""
    
    AVAILABLE_MODELS = [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
    ]
    
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


class PerplexityClient(BaseLLMClient):
    """
    Client for Perplexity API.
    Great for search-augmented responses.
    """
    
    AVAILABLE_MODELS = [
        "sonar-reasoning-pro",
        "sonar-reasoning",
        "sonar-pro",
        "sonar",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-small-128k-online",
    ]
    
    def __init__(
        self,
        model: str = "sonar-pro",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai"
        
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
        """Get completion from Perplexity."""
        
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
            logger.error(f"Perplexity API error: {str(e)}")
            raise


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama (local models).
    Run models locally - completely free!
    """
    
    AVAILABLE_MODELS = [
        "llama3.3",
        "llama3.2",
        "llama3.1",
        "mistral",
        "mixtral",
        "qwen2.5",
        "gemma2",
        "phi4",
        "deepseek-coder-v2",
    ]
    
    def __init__(
        self,
        model: str = "llama3.3",
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
        
        if messages is None:
            messages = []
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
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
            payload["messages"].insert(0, {"role": "system", "content": system})
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
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
        
        if system is None and ("chunk" in prompt_lower or "section" in prompt_lower or "documents" in prompt_lower):
            return "I found relevant information in this chunk. The key points are: [simulated content analysis mentioning Ada Lovelace and Alan Turing]"
        
        elif "summarize" in prompt_lower:
            return "Summary: [simulated summary of the provided text]"
        
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
    """Factory for creating LLM clients with ALL latest 2025 models."""
    
    # Complete provider information with ALL latest models
    PROVIDER_INFO = {
        "openai": {
            "name": "OpenAI",
            "description": "GPT-5, GPT-4.5, o3, o4-mini, o1, GPT-4o",
            "models": OpenAIClient.AVAILABLE_MODELS,
            "key_url": "https://platform.openai.com/api-keys",
            "key_prefix": "sk-",
            "cost": "$0.15-30 per 1M tokens",
            "recommended": ["gpt-4o-mini", "o4-mini", "gpt-4o"]
        },
        "anthropic": {
            "name": "Anthropic",
            "description": "Claude Opus 4.6, Sonnet 4.5, Haiku 4.5, Claude 3.5",
            "models": AnthropicClient.AVAILABLE_MODELS,
            "key_url": "https://console.anthropic.com/",
            "key_prefix": "sk-ant-",
            "cost": "$0.50-30 per 1M tokens",
            "recommended": ["claude-3-5-sonnet-20241022", "claude-sonnet-4-5", "claude-haiku-4-5"]
        },
        "google": {
            "name": "Google Gemini",
            "description": "Gemini 3.0, 2.5, 2.0 Pro/Flash/Lite",
            "models": GeminiClient.AVAILABLE_MODELS,
            "key_url": "https://aistudio.google.com/app/apikey",
            "key_prefix": "",
            "cost": "$0.08-10 per 1M tokens",
            "recommended": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
        },
        "kimi": {
            "name": "Moonshot AI (Kimi)",
            "description": "Kimi K3, K2.5, k1.5 - China's best models",
            "models": KimiClient.AVAILABLE_MODELS,
            "key_url": "https://platform.moonshot.cn/",
            "key_prefix": "sk-",
            "cost": "$0.50-5 per 1M tokens",
            "recommended": ["kimi-k2.5", "kimi-k3", "kimi-k1.5"]
        },
        "groq": {
            "name": "Groq",
            "description": "Llama 4, 3.3, 3.1, Mixtral - FASTEST inference",
            "models": GroqClient.AVAILABLE_MODELS,
            "key_url": "https://console.groq.com/keys",
            "key_prefix": "gsk_",
            "cost": "$0.05-0.60 per 1M tokens",
            "recommended": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-maverick-17b-128e-instruct"]
        },
        "together": {
            "name": "Together AI",
            "description": "Llama 4, DeepSeek V3, Qwen 3, Mixtral",
            "models": TogetherClient.AVAILABLE_MODELS,
            "key_url": "https://api.together.xyz/",
            "key_prefix": "",
            "cost": "$0.18-5 per 1M tokens",
            "recommended": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3"]
        },
        "deepseek": {
            "name": "DeepSeek",
            "description": "DeepSeek V3, R1, Coder - CHEAPEST",
            "models": DeepSeekClient.AVAILABLE_MODELS,
            "key_url": "https://platform.deepseek.com/",
            "key_prefix": "sk-",
            "cost": "$0.07-0.30 per 1M tokens",
            "recommended": ["deepseek-chat", "deepseek-coder"]
        },
        "mistral": {
            "name": "Mistral AI",
            "description": "Mistral Large, Medium, Small, Codestral",
            "models": MistralClient.AVAILABLE_MODELS,
            "key_url": "https://console.mistral.ai/",
            "key_prefix": "",
            "cost": "$0.20-8 per 1M tokens",
            "recommended": ["mistral-large-latest", "mistral-small-latest"]
        },
        "cohere": {
            "name": "Cohere",
            "description": "Command R+, Command R, Command",
            "models": CohereClient.AVAILABLE_MODELS,
            "key_url": "https://dashboard.cohere.com/api-keys",
            "key_prefix": "",
            "cost": "$0.50-2.50 per 1M tokens",
            "recommended": ["command-r-plus", "command-r"]
        },
        "perplexity": {
            "name": "Perplexity",
            "description": "Sonar Pro, Sonar - Search-augmented",
            "models": PerplexityClient.AVAILABLE_MODELS,
            "key_url": "https://www.perplexity.ai/settings/api",
            "key_prefix": "pplx-",
            "cost": "$0.50-5 per 1M tokens",
            "recommended": ["sonar-pro", "sonar"]
        },
        "azure": {
            "name": "Azure OpenAI",
            "description": "Enterprise OpenAI (GPT-4o, o1, etc.)",
            "models": ["gpt-4o", "gpt-4", "gpt-35-turbo", "o1", "o3-mini"],
            "key_url": "https://portal.azure.com",
            "key_prefix": "",
            "cost": "Enterprise pricing",
            "recommended": ["gpt-4o", "o1"]
        },
        "ollama": {
            "name": "Ollama (Local)",
            "description": "Run Llama, Mistral, Qwen locally - FREE",
            "models": OllamaClient.AVAILABLE_MODELS,
            "key_url": "https://ollama.com/",
            "key_prefix": "N/A",
            "cost": "FREE (runs on your machine)",
            "recommended": ["llama3.3", "mistral", "qwen2.5"]
        },
        "mock": {
            "name": "Mock (Testing)",
            "description": "Fake responses for testing - FREE",
            "models": ["mock-model"],
            "key_url": "N/A",
            "key_prefix": "N/A",
            "cost": "FREE",
            "recommended": ["mock-model"]
        }
    }
    
    @staticmethod
    def get_provider_info() -> Dict[str, Dict]:
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
        
        elif provider == "anthropic":
            model = model or "claude-3-5-sonnet-20241022"
            return AnthropicClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "google" or provider == "gemini":
            model = model or "gemini-2.5-flash"
            return GeminiClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "kimi" or provider == "moonshot":
            model = model or "kimi-k2.5"
            return KimiClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "groq":
            model = model or "llama-3.3-70b-versatile"
            return GroqClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "together":
            model = model or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            return TogetherClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "deepseek":
            model = model or "deepseek-chat"
            return DeepSeekClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "mistral":
            model = model or "mistral-large-latest"
            return MistralClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "cohere":
            model = model or "command-r-plus"
            return CohereClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "perplexity":
            model = model or "sonar-pro"
            return PerplexityClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "azure":
            model = model or "gpt-4o"
            return AzureOpenAIClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "ollama":
            model = model or "llama3.3"
            return OllamaClient(model=model, **kwargs)
        
        elif provider == "mock":
            return MockLLMClient(model=model or "mock-model", **kwargs)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
