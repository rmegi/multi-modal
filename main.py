import cv2
from utils.ollama_handler import OllamaHandler


local_ai_agent = OllamaHandler(
    model="gemma3",
    base_url="http://localhost:11434",
)

image_path = "image.png"


def main():
    while True:
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

        response = local_ai_agent.ask(prompt='describe what you see', image_path=image_path)
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: Sorry, I couldn't process that.")


if __name__ == "__main__":
    main()
