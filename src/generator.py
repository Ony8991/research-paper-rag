import os
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are an expert research assistant. "
        "Answer the following question based ONLY on the provided context. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


class OllamaGenerator:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


class GroqGenerator:
    """Groq Cloud — free tier, fast inference. Get a key at console.groq.com"""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        r = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class HuggingFaceGenerator:
    """HuggingFace Inference API — most models require a Pro account."""

    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model = os.getenv("HF_GENERATION_MODEL", "HuggingFaceH4/zephyr-7b-beta")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }
        r = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 404:
            raise RuntimeError(
                f"Model '{self.model}' is gated or not found (requires HF Pro). "
                "Use Groq instead: set GROQ_API_KEY in .env"
            )
        if r.status_code == 503:
            raise RuntimeError("HuggingFace model is loading, retry in 20s.")
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return str(result)


class Generator:
    def __init__(self):
        self._ollama = OllamaGenerator(
            model=os.getenv("GENERATION_MODEL", "mistral:7b-instruct")
        )
        self._groq = GroqGenerator()
        self._hf = HuggingFaceGenerator()
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        forced = os.getenv("GENERATOR_BACKEND", "").lower()
        if forced in ("ollama", "groq", "huggingface", "none"):
            return forced
        if self._ollama.is_available():
            return "ollama"
        if self._groq.is_available():
            return "groq"
        if self._hf.is_available():
            return "huggingface"
        return "none"

    def generate(self, question: str, context_chunks: List[str]) -> Optional[str]:
        if self.backend == "none" or not context_chunks:
            return None
        prompt = _build_prompt(question, context_chunks)
        if self.backend == "ollama":
            return self._ollama.generate(prompt)
        if self.backend == "groq":
            return self._groq.generate(prompt)
        if self.backend == "huggingface":
            return self._hf.generate(prompt)
        return None
