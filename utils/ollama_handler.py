import httpx
import base64
import os
import time
import hashlib
import ollama
import json
from utils.prompts import gemma3_12b_prompt


MAX_IMAGES = 2


class OllamaHandler:
    def __init__(self, model="gemma3:12b", base_url=None):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://192.168.68.201:11434"
        )
        self.chat_url = f"{self.base_url}/api/chat"
        self.clear_chat()
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
        self.messages_history.append(user_message)
        answer = None
        try:
            start_time = time.perf_counter()
            response = ollama.chat(model=self.model, options={'temperature':temperature}, format="json", messages=self.messages_history + [user_message])
            duration = time.perf_counter() - start_time
            print(f"✅ Ollama responded in {duration:.2f}s")
            answer = json.loads(response.get("message", {}).get("content", ""))
            self.messages_history.append({"role": "assistant", "content": json.dumps(answer, indent=2)})
            print([m.keys() for m in self.messages_history])
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            answer = str(e)
        
        if sum([len(m.get("images", [])) for m in self.messages_history]) > MAX_IMAGES:
            img_cnt = 0
            self.messages_history.reverse()
            for m in self.messages_history:
                if "images" in m and img_cnt > MAX_IMAGES:
                    del m["images"]
                img_cnt += len(m.get("images", []))
            self.messages_history.reverse()

        return answer
