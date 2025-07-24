import json
from pathlib import Path


class PromptManager:
    def __init__(self, prompt_path="prompt.json"):
        self.prompt_file = Path(prompt_path)
        self.prompts = []
        self.load_prompt()

    def load_prompt(self):
        """Load classes from JSON."""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"{self.prompt_file} does not exist.")

        with self.prompt_file.open("r") as f:
            data = json.load(f)

        self.prompts = data.get("classes", [])
        if not self.prompts:
            raise ValueError("No classes defined in prompt.json")

        print(f"✅ Prompt loaded: {self.prompts}")

    def update_prompt(self, new_classes: list[str]):
        """Update class list and save to JSON."""
        self.prompts = new_classes
        with self.prompt_file.open("w") as f:
            json.dump({"classes": new_classes}, f, indent=2)
        print(f"🔄 Prompt updated: {new_classes}")

    def get_prompts(self):
        return self.prompts
