"""Example client for the local Qwen3-4B server (OpenAI-compatible API).

Run the server first:  python serve_qwen3.py
Then:                 python client_example.py
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def chat(prompt: str, system: str = None, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "qwen3-4b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # 1. plain completion
    print(chat("用一句话说明什么是端到端自动驾驶。"))

    # 2. with a system prompt (useful for reward shaping / labeling tasks)
    print(
        chat(
            "场景：自车在高速上以80km/h行驶，前方20米有一辆慢车切入。\n"
            "请给出3条层次化子目标奖励。",
            system="你是自动驾驶强化学习专家，输出必须简洁的要点列表。",
        )
    )
