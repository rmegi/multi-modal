A simple handler for interacting with [Ollama](https://ollama.com/) models via HTTP API. This project provides utilities to send prompts, manage models, and process responses from Ollama in your applications.

## Features

- Send prompts and receive responses from Ollama models
- Manage and list available models
- Easy integration with Python projects

## Installation

```bash
pip install ollama-handler
```

## Usage

```python
from ollama_handler import OllamaClient

client = OllamaClient(base_url="http://localhost:11434")
response = client.send_prompt("llama2", "Hello, how are you?")
print(response)
```

## Requirements

- Python 3.7+
- Ollama running locally or accessible via network
