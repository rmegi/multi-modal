import httpx
import base64
import os
import time
import hashlib
from utils.prompts import gemma3_12b_prompt


class OllamaHandler:
    def __init__(self, model="llava:latest", base_url=None):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://192.168.68.201:11434"
        )
        self.chat_url = f"{self.base_url}/api/chat"
        self.messages_history = [{"role": "system", "content": gemma3_12b_prompt}]
        self._last_image_hash = None

    def _hash_image(self, image_bytes):
        return hashlib.sha256(image_bytes).hexdigest()

    def clear_chat(self):
        self.messages_history = [{"role": "system", "content": gemma3_12b_prompt}]
        return "Chat history cleared."

    def get_chat_history(self):
        return self.messages_history

    def ask(self, prompt, image_path=None, temperature=0.1):
        user_message = {"role": "user", "content": prompt}

        # Add image if provided
        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                    current_hash = self._hash_image(image_bytes)

                    if self._last_image_hash == current_hash:
                        print("⚠️ Same image as last time (hash unchanged)")
                    else:
                        print("✅ New image detected")
                        self._last_image_hash = current_hash

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
        }

        try:
            start_time = time.perf_counter()
            response = httpx.post(
                self.chat_url, json=payload, timeout=20.0  # hard timeout
            )
            response.raise_for_status()
            duration = time.perf_counter() - start_time
            print(f"✅ Ollama responded in {duration:.2f}s")

            result = response.json()
            answer = result.get("message", {}).get("content", "")

            self.messages_history.append({"role": "user", "content": prompt})
            self.messages_history.append({"role": "assistant", "content": answer})

            return answer

        except httpx.TimeoutException:
            print("❌ Ollama request timed out (>20s).")
            return None
        except httpx.HTTPStatusError as e:
            print(f"❌ Ollama HTTP error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
