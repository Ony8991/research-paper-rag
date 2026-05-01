import os
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "Tu es un assistant expert en recherche scientifique. "
        "Réponds à la question suivante en te basant UNIQUEMENT sur le contexte fourni. "
        "Si la réponse n'est pas dans le contexte, dis-le clairement.\n\n"
        f"CONTEXTE:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "RÉPONSE:"
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


class HuggingFaceGenerator:
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model = model
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"

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
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()
        return str(result)


class Generator:
    def __init__(self):
        ollama_model = os.getenv("GENERATION_MODEL", "mistral:7b-instruct")
        hf_model = os.getenv("HF_GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

        self._ollama = OllamaGenerator(model=ollama_model)
        self._hf = HuggingFaceGenerator(model=hf_model)
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        forced = os.getenv("GENERATOR_BACKEND", "").lower()
        if forced in ("ollama", "huggingface", "none"):
            return forced
        if self._ollama.is_available():
            return "ollama"
        if self._hf.is_available():
            return "huggingface"
        return "none"

    def generate(self, question: str, context_chunks: List[str]) -> Optional[str]:
        if self.backend == "none" or not context_chunks:
            return None
        prompt = _build_prompt(question, context_chunks)
        if self.backend == "ollama":
            return self._ollama.generate(prompt)
        if self.backend == "huggingface":
            return self._hf.generate(prompt)
        return None
