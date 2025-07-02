import base64
import os
import time
from pydantic import BaseModel
from typing import Optional
from utils.prompts import vision_prompt_v1
from ollama import chat
import ollama._client  # Needed to override base URL


class OllamaHandler:
    def __init__(self, model="llava:latest", base_url=None):
        self.model = model
        self.messages_history = [{"role": "system", "content": vision_prompt_v1}]
        self._last_image_hash = None

        if base_url:
            ollama._client.BASE_URL = base_url  # Override internal base URL

    def get_chat_history(self):
        return self.messages_history

    def ask(
        self, prompt: str, image_path: Optional[str] = None, temperature: float = 0
    ):
        user_message = {"role": "user", "content": prompt}

        if image_path:
            user_message["images"] = [image_path]  # raw path, not base64

        try:
            start_time = time.perf_counter()

            response = chat(
                model=self.model,
                messages=self.messages_history + [user_message],
                options={"temperature": temperature},
            )

            duration = time.perf_counter() - start_time
            print(f"✅ Ollama responded in {duration:.2f}s")

            content = response.message.content
            self.messages_history.append({"role": "user", "content": prompt})
            self.messages_history.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            print(f"❌ Error while calling Ollama chat: {e}")
            return None
