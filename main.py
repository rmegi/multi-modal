from utils.ollama_handler import OllamaHandler
import cv2


local_ai_agent = OllamaHandler(
    model="gemma3:12b",
    base_url="http://192.168.68.201:11434",
    # base_url="http://localhost:11434"
)

image_path = "image.png"


def main():
    print("Welcome to the Ollama Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        # Capture the current image from the webcam.
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('image.png', frame)
            else:
                print("Failed to capture an image.")
            cap.release()
        except Exception as e:
            print(f'Using default image ({e})')

        response = local_ai_agent.ask(prompt=user_input, image_path=image_path)
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: Sorry, I couldn't process that.")


if __name__ == "__main__":
    main()
