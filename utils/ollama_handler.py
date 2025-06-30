import base64
import os
import time
import hashlib
from ollama import Client
import json
from utils.prompts import gemma3_12b_prompt


MAX_IMAGES = 1


class OllamaHandler:
    def __init__(self, model="gemma3:12b", base_url=None):
        self.model = model
        base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://192.168.68.201:11434"
        )
        self._client = Client(host=base_url)
        self.clear_chat()
        self._last_image_hash = None

    def _hash_image(self, image_bytes):
        return hashlib.sha256(image_bytes).hexdigest()

    def clear_chat(self):
        self.messages_history = [{"role": "system", "content": "You are a scene analyzer. You recive an image and describe what you see in the image. Be short and precise."}]
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

        img_cnt = 0
        for i in range(len(self.messages_history)-1, 0, -1):
            img_cnt += len(self.messages_history[i].get("images", []))
            if "images" in self.messages_history[i] and img_cnt > MAX_IMAGES:
                del self.messages_history[i]["images"]

        answer = None
        try:
            start_time = time.perf_counter()
            response = self._client.chat(model=self.model, options={'temperature':temperature}, format="json", messages=self.messages_history + [user_message])
            duration = time.perf_counter() - start_time
            print(f"✅ Ollama responded in {duration:.2f}s")
            answer = json.loads(response.get("message", {}).get("content", ""))
            self.messages_history.append({"role": "assistant", "content": json.dumps(answer, indent=2)})
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            answer = str(e)

        return answer
