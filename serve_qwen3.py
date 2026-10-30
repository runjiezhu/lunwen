"""Local Qwen3-4B serving on RTX 5060 Ti (8GB).

Runs the model with bitsandbytes 4-bit NF4 quantization so it fits in 8GB VRAM,
and exposes an OpenAI-compatible /v1/chat/completions endpoint on port 8000.

Usage:
    python serve_qwen3.py
    curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" ^
      -d "{\"model\":\"qwen3-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
"""

import time
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"C:\Users\zhuru\yuelaogame\lunwen\models\Qwen3-4B-Instruct-2507"
HOST = "127.0.0.1"
PORT = 8000

app = FastAPI(title="Qwen3-4B local server")

tokenizer = None
model = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-4b"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = False


def load_model() -> None:
    global tokenizer, model
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Loading model (4-bit NF4)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        },
    )
    model.eval()
    if model.generation_config is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    print("Model ready.", flush=True)


def generate(messages, max_tokens, temperature, top_p, top_k) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen3-4b", "loaded": model is not None}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if model is None:
        raise RuntimeError("Model not loaded")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    start = time.time()
    text = generate(
        messages, req.max_tokens, req.temperature, req.top_p, req.top_k
    )
    prompt_tokens = len(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    )
    completion_tokens = len(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "qwen3-4b", "object": "model"}]}


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host=HOST, port=PORT)
