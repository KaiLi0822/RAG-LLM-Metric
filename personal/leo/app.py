import time
import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

max_memory = {i: "12GB" for i in range(1)}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="sequential",      
    torch_dtype=torch.bfloat16,
    max_memory=max_memory,
    attn_implementation="eager"
)

model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

chain_of_thought_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 10000
    temperature: float = 1.0
    top_p: float = 1.0


@app.get("/health")
def health_check():
    """
    Health check endpoint: returns a simple JSON indicating the service is running.
    """
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Emulates the OpenAI ChatCompletion API for your custom model.
    """
    messages = [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    input_tensor = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    generation_output = model.generate(
        input_tensor.to(model.device),
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    raw_output = tokenizer.decode(
        generation_output[0][input_tensor.shape[1]:], 
        skip_special_tokens=True
    )
    # Remove chain-of-thought
    cleaned_output = chain_of_thought_pattern.sub("", raw_output).strip()

    response_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    prompt_tokens = input_tensor.shape[1]
    completion_tokens = generation_output.shape[1] - input_tensor.shape[1]
    total_tokens = prompt_tokens + completion_tokens

    response_body = {
        "id": response_id,
        "object": "chat.completion",
        "created": created_time,
        "model": request.model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": cleaned_output
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }

    return response_body


