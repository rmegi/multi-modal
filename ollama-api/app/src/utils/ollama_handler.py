import json
import httpx
import base64
import os
import time
from pydantic import BaseModel
from utils.prompts import gemma3_12b_prompt_v2 as gemma_prompt


class OllamaResponse(BaseModel):
    description: str
    threats: list[str]


class OllamaHandler:
    def __init__(self, model="gemma3:12b", base_url=None):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://192.168.68.201:11434"
        )
        self.chat_url = f"{self.base_url}/api/chat"
        self.messages_history = [{"role": "system", "content": gemma_prompt}]
        self.format = OllamaResponse.model_json_schema()

    def get_chat_history(self):
        return self.messages_history

    def ask(self, prompt, image_path=None, temperature=0) -> OllamaResponse:
        user_message = {"role": "user", "content": prompt}

        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    user_message["images"] = [base64_image]
            except Exception as e:
                print(f"❌ Failed to read image: {e}")
                return None

        payload = {
            "model": self.model,
            "messages": self.messages_history + [user_message],
            "stream": False,
            "temperature": temperature,
            "keep_alive": "10m",
            "format": self.format,
        }

        try:
            start_time = time.perf_counter()
            response = httpx.post(
                self.chat_url, json=payload, timeout=20.0
            )  # hard timeout
            response.raise_for_status()
            duration = time.perf_counter() - start_time
            print(f"✅ Ollama responded in {duration:.2f}s")

            result = response.json().get("message", {}).get("content", "")

            try:
                result_json = json.loads(result)
                description = result_json.get("description", "")
                threats = result_json.get("threats", [])
            except json.JSONDecodeError:
                print("❌ Failed to parse description JSON.")
                description = ""
                threats = []

            return OllamaResponse(description=description, threats=threats)

        except httpx.TimeoutException:
            print("❌ Ollama request timed out (>20s).")
            return None
        except httpx.HTTPStatusError as e:
            print(f"❌ Ollama HTTP error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
