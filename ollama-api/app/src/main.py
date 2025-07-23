from urllib import response

import requests
from utils.ollama_handler import OllamaHandler
from utils.utils import parse_response
import time

local_ai_agent = OllamaHandler(
    model="gemma3:12b", base_url="http://192.168.68.201:11434"
)

image_path = "bomb3.png"


def main():
    with open(image_path, "rb") as f:
        files = {"file": f}
        start_time = time.time()
        response = requests.post("http://100.94.195.34:9000/detect", files=files)
        elapsed_time = time.time() - start_time
        print(f"Request took {elapsed_time:.2f} seconds")
    print(response.json())

    # print("Welcome to the Ollama Chatbot!")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         print("Exiting the chat. Goodbye!")
    #         break

    #     response = local_ai_agent.ask(prompt=user_input, image_path=image_path)
    #     if response:
    #         print(f"Bot: Description: {response.description}")
    #         print(f"Bot: detected: {response.detected}")
    #     else:
    #         print("Bro: Sorry bro, I couldn't process that.")


if __name__ == "__main__":
    main()
