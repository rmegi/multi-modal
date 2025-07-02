from utils.ollama_handler import OllamaHandler


local_ai_agent = OllamaHandler(
    model="gemma3:12b", base_url="http://192.168.68.201:11434"
)

image_path = "image.png"  # Path to the image file, if needed


def main():
    print("Welcome to the Ollama Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        response = local_ai_agent.ask(prompt=user_input, image_path=image_path)
        if response:
            print(f"{local_ai_agent.model}: {response}")
        else:
            print("Bro: Sorry bro, I couldn't process that.")


if __name__ == "__main__":
    main()
