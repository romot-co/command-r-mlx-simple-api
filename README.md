# Command-R MLX API Server

## Overview

This repository contains the Flask application for a machine learning model serving API. It provides several endpoints to generate text, simulate chat conversations, tool executions, and grounded responses based on different input parameters.

### Features

- Generate text with a given prompt and additional parameters.
- Simulate chat conversations and generate appropriate responses.
- Execute tools with specific arguments in a conversational context.
- Retrieve grounded responses using provided documents and conversation history.

## Installation

To run this application, you need Python 3 and pip installed on your system. Follow these steps to set up the server:

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the server:

    ```sh
    python server.py --port 5000 --model 'mlx-community/c4ai-command-r-v01-4bit' --debug
    ```

Replace `<repository-url>` and `<repository-folder>` with the actual URL and folder name of your cloned repository.

## API Endpoints

### Generate Text (`/generate`)

Generates text based on the provided prompt.

#### POST Request Body

```json
{
    "prompt": "Enter your prompt here",
    "temperature": 0.2,
    "max_tokens": 131072
}
```

#### Response

```json
{
    "generated_text": "Generated text will be here."
}
```

### Chat (`/chat`)

Simulates a chat conversation and generates a response.

#### POST Request Body

```json
{
    "conversation": [
        {"role": "user", "content": "User's message"},
        {"role": "assistant", "content": "Assistant's response"}
    ],
    "temperature": 0.2,
    "max_tokens": 131072
}
```

#### Response

```json
{
    "generated_text": "Generated chat response will be here."
}
```

### Use Tool (`/tool`)

Runs a specified tool within the conversation context.

#### POST Request Body

```json
{
    "conversation": [
        {"role": "user", "content": "User's message"},
        {"role": "assistant", "content": "Assistant's response"}
    ],
    "tools": [
        {
            "name": "internet_search",
            "description": "Searches the internet for the given query.",
            "parameter_definitions": {
                "query": {
                    "description": "Query to search the internet with",
                    "type": "str",
                    "required": true
                }
            }
        }
    ]
}
```

#### Response

```json
{
    "tool_response": "Generated tool response will be here."
}
```

### Grounded Response Generation (`/rag`)

Generates grounded responses based on the conversation and supplemental documents.

#### POST Request Body

```json
{
    "conversation": [
        {"role": "user", "content": "User's message"},
        {"role": "assistant", "content": "Assistant's response"}
    ],
    "documents": [
        { "title": "Document Title", "text": "Some relevant information." }
    ],
    "citation_mode": "accurate"
}
```

#### Response

```json
{
    "rag_response": "Generated grounded response will be here."
}
```

## Usage

1. Ensure that the Flask server is running as instructed above.
2. Utilize tools such as `curl` or Postman to send HTTP POST requests to the desired endpoint with the required JSON payload.

## Notes

- Make sure to validate input parameters and catch exceptions as per the example code.
- Always check for the latest updates and documentation.

---

Remember to replace placeholder texts with actual values suited to your environment and use case.