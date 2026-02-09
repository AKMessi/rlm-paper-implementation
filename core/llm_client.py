"""
LLM Client for RLM
==================

Unified interface for different LLM providers (OpenAI, Anthropic, etc.)
Supports both root and sub-LM calls.
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
        
        # Build messages
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system:
            # Insert system message at the beginning
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
        
        # Build messages
        if messages is None:
            messages = []
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
        # Convert to Anthropic format (no system in messages)
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


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing.
    Simulates responses for development without API costs.
    """
    
    def __init__(self, model: str = "mock-model", **kwargs):
        super().__init__(model, **kwargs)
        self.call_count = 0
        self.is_root = True  # Assume root by default
        
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
        
        # Check if this is a sub-LM call by looking for chunk/section analysis
        prompt_text = prompt or ""
        if messages:
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    prompt_text += msg["content"]
        
        prompt_lower = prompt_text.lower()
        
        # Sub-LM calls don't have system prompts with REPL
        if system is None and ("chunk" in prompt_lower or "section" in prompt_lower or "documents" in prompt_lower):
            return "I found relevant information in this chunk. The key points are: [simulated content analysis mentioning Ada Lovelace and Alan Turing]"
        
        elif "summarize" in prompt_lower:
            return "Summary: [simulated summary of the provided text]"
        
        # Root RLM - track iterations and provide final answer
        if self.call_count <= 2:
            # First calls: explore context
            return '''```repl
# Initial exploration of the context
print("Exploring context...")
print(f"Type: {type(context)}")
print(f"Length: {len(context) if hasattr(context, '__len__') else 'N/A'}")

if isinstance(context, list):
    print(f"Number of items: {len(context)}")
    if len(context) > 0:
        print(f"First item preview: {str(context[0])[:200]}")
        
# Look for section 50
if isinstance(context, list) and len(context) >= 50:
    section_50 = context[49]
    print(f"Section 50 preview: {section_50[:300]}")
```
'''
        else:
            # Provide final answer
            return '''Based on my analysis of the document, I found the relevant information in section 50.

FINAL(Section 50 discusses The History of Computing, mentioning Ada Lovelace, Alan Turing, mechanical calculators, and the ENIAC computer.)'''
        


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create(
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client based on provider.
        
        Args:
            provider: "openai", "anthropic", or "mock"
            model: Model name (optional, uses default if not specified)
            api_key: API key (optional, uses env var if not specified)
            **kwargs: Additional parameters
            
        Returns:
            Configured LLM client
        """
        provider = provider.lower()
        
        if provider == "openai":
            model = model or "gpt-4o-mini"
            return OpenAIClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "anthropic":
            model = model or "claude-3-5-sonnet-20241022"
            return AnthropicClient(model=model, api_key=api_key, **kwargs)
        
        elif provider == "mock":
            return MockLLMClient(model=model or "mock-model", **kwargs)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
