from abc import ABC, abstractmethod
import os
import re
from openai import OpenAI
import requests
import json

class LLMClient(ABC):
    """Base class for LLM clients with standardized invocation interface"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Execute LLM call with given prompt and return response text
        
        Args:
            prompt: Input text/prompt for the LLM
            
        Returns:
            Generated text response from LLM
        """
        pass

class LocalDeepSeekR1(LLMClient):
    """using local deepSeek distill Qwen with OpenAI-compatible client
       Follows instruction with https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations
    """

    def __init__(self,
                 model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                 base_url = "http://127.0.0.1:30000/v1",
                 **kwargs):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable required")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.params = {
            "temperature": 0.6,
            "max_tokens": 32000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.5,
        }
        self.params.update(kwargs)        

    def generate(self, prompt: str) -> str:
        """Execute synchronous LLM call"""
        messages = [
            {"role": "user", "content": f"{prompt} \n\nAssistant: <think>\n"}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.params
        )
        match  = re.search(r'</think>\n\n(.*)', completion.choices[0].message.content, re.DOTALL)
        return match.group(1)


class OpenAIClientLLM(LLMClient):
    """Concrete implementation using OpenAI-compatible client"""
    
    def __init__(self, 
                 model: str = "meta-llama/Llama-3.3-70B-Instruct",
                 system_message: str = "You are a helpful assistant",
                 base_url: str = "https://api-eu.centml.com/openai/v1",
                 **kwargs):
        """
        Initialize OpenAI-style client
        
        Args:
            model: Model identifier string
            system_message: System prompt for conversation context
            base_url: API endpoint URL
            kwargs: Additional parameters for completions
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_message = system_message
        self.params = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.5,
        }
        self.params.update(kwargs)

    def generate(self, prompt: str) -> str:
        """Execute synchronous LLM call"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.params
        )
        
        return completion.choices[0].message.content

class HTTPLLM(LLMClient):
    """Concrete implementation using generic HTTP API endpoint"""
    
    def __init__(self,
                 model: str = "deepseek_r1",
                 base_url: str = "https://cloud.luchentech.com/api/maas/chat/completions",
                 system_message: str = "You are a helpful and harmless assistant. You should think step-by-step.",
                 **kwargs):
        """
        Initialize HTTP client
        
        Args:
            model: Model identifier string
            base_url: API endpoint URL
            system_message: System prompt for conversation context  
            kwargs: Additional parameters for completions
        """
        api_key = os.getenv("MAAS_API_KEY")
        if not api_key:
            raise ValueError("MAAS_API_KEY environment variable required")
            
        self.model = model
        self.base_url = base_url
        self.system_message = system_message
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.params = {
            "stream": False,
            "max_tokens": 32000,
            **kwargs
        }

    def generate(self, prompt: str) -> str:
        """Execute synchronous HTTP request"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            **self.params
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

# Example usage
if __name__ == "__main__":
    # Set environment variables first
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    os.environ["MAAS_API_KEY"] = "your-api-key-here"

    # OpenAI client example
    openai_llm = OpenAIClientLLM()
    print("OpenAI response:", openai_llm.generate("Hello world!"))
    
    # HTTP client example
    http_llm = HTTPLLM()
    print("HTTP response:", http_llm.generate("Explain quantum computing in 3 sentences"))