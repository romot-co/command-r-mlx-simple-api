import argparse
from flask import Flask, request, jsonify
from flask.logging import create_logger
from mlx_lm import load, generate

app = Flask(__name__)
logger = create_logger(app)

def validate_temperature(temperature):
    if not (0.0 <= temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

def validate_max_tokens(max_tokens):
    if not (1 <= max_tokens <= 131072):
        raise ValueError("Max tokens must be between 1 and 131072")

def validate_citation_mode(citation_mode):
    if citation_mode not in ['fast', 'accurate']:
        raise ValueError("Citation mode must be either 'fast' or 'accurate'")

def validate_tools(tools):
    for tool in tools:
        if 'name' not in tool:
            raise ValueError("Each tool must have a 'name' field")
        if 'description' not in tool:
            raise ValueError("Each tool must have a 'description' field")
        if 'parameter_definitions' not in tool:
            raise ValueError("Each tool must have a 'parameter_definitions' field")
        for param_name, param_def in tool['parameter_definitions'].items():
            if 'description' not in param_def:
                raise ValueError(f"Parameter '{param_name}' must have a 'description' field")
            if 'type' not in param_def:
                raise ValueError(f"Parameter '{param_name}' must have a 'type' field")
            if 'required' not in param_def:
                raise ValueError(f"Parameter '{param_name}' must have a 'required' field")

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Generate text based on the given prompt.

    Request JSON:
    {
        "prompt": "The prompt to generate text from",
        "temperature": (optional, default=0.2) The temperature for text generation (0.0 to 1.0),
        "max_tokens": (optional, default=131072) The maximum number of tokens to generate (1 to 131072)
    }

    Response JSON:
    {
        "generated_text": "The generated text"
    }
    """

    logger.info("Received a generation request.")

    try:
        data = request.get_json()
        prompt = data['prompt']
        temperature = data.get('temperature', 0.2)
        max_tokens = data.get('max_tokens', 131072)
        
        validate_temperature(temperature)
        validate_max_tokens(max_tokens)
        
        response = generate(
            model, 
            tokenizer,
            prompt=prompt,
            verbose=True,
            temp=temperature,
            max_tokens=max_tokens,
        )

        logger.debug(f"Generated response: {response}")
        return jsonify({"generated_text": response})
    
    except KeyError as e:
        logger.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({"error": f"Missing key in request JSON: {str(e)}"}), 400
    
    except ValueError as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Generate a chat response based on the given conversation.

    Request JSON:
    {
        "conversation": [
            {"role": "user", "content": "User's message"},
            {"role": "assistant", "content": "Assistant's response"},
            ...
        ],
        "temperature": (optional, default=0.2) The temperature for response generation (0.0 to 1.0),
        "max_tokens": (optional, default=131072) The maximum number of tokens to generate (1 to 131072)
    }

    Response JSON:
    {
        "generated_text": "The generated chat response"
    }
    """
    try:
        data = request.get_json()
        conversation = data['conversation']
        temperature = data.get('temperature', 0.2)
        max_tokens = data.get('max_tokens', 131072)
        
        validate_temperature(temperature)
        validate_max_tokens(max_tokens)
        
        inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        response = generate(
            model, 
            tokenizer,
            prompt=inputs,
            verbose=True,
            temp=temperature,
            max_tokens=max_tokens,
        )

        return jsonify({"generated_text": response})
    
    except KeyError as e:
        return jsonify({"error": f"Missing key in request JSON: {str(e)}"}), 400
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/tool', methods=['POST'])
def use_tool():
    """
    Generate a tool response based on the given conversation and tools.

    Request JSON:
    {
        "conversation": [
            {"role": "user", "content": "User's message"},
            {"role": "assistant", "content": "Assistant's response"},
            ...
        ],
        "tools": [
            {
                "name": "internet_search",
                "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
                "parameter_definitions": {
                "query": {
                    "description": "Query to search the internet with",
                    "type": 'str',
                    "required": True
                },
            },
            ...
        ]
    }

    Response JSON:
    {
        "tool_response": "The generated tool response"
    }
    """

    logger.info("Received a tool request.")

    try:
        data = request.get_json()
        conversation = data['conversation']
        tools = data['tools']
        
        validate_tools(tools)
        
        formatted_input = tokenizer.apply_tool_use_template(conversation, tools=tools, tokenize=False, add_generation_prompt=True)
        
        response = generate(model, tokenizer, prompt=formatted_input, verbose=True)

        return jsonify({"tool_response": response})
    
    except KeyError as e:
        logger.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({"error": f"Missing key in request JSON: {str(e)}"}), 400
    
    except ValueError as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/rag', methods=['POST'])
def rag():
    """
    Generate a grounded response based on the given conversation and documents.

    Request JSON:
    {
        "conversation": [
            {"role": "user", "content": "User's message"},
            {"role": "assistant", "content": "Assistant's response"},
            ...
        ],
        "documents": [
            { "title": "Tall penguins", "text": "Emperor penguins are the tallest growing up to 122 cm in height." }, 
            { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
        ],
        "citation_mode": (optional, default="accurate") The citation mode ("fast" or "accurate")
    }

    Response JSON:
    {
        "rag_response": "The generated grounded response"
    }
    """

    logger.info("Received a RAG request.")

    try:
        data = request.get_json()
        conversation = data['conversation']
        documents = data['documents']
        citation_mode = data.get('citation_mode', 'accurate')
        
        validate_citation_mode(citation_mode)
        
        formatted_input = tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            citation_mode=citation_mode,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        response = generate(model, tokenizer, prompt=formatted_input, verbose=True)

        return jsonify({"rag_response": response})
    
    except KeyError as e:
        logger.error(f"Missing key in request JSON: {str(e)}")
        return jsonify({"error": f"Missing key in request JSON: {str(e)}"}), 400
    
    except ValueError as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Command-R MLX API server')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Port number for the server')
    parser.add_argument('--model', '-m', type=str, default='mlx-community/c4ai-command-r-v01-4bit', help='Model name')
    parser.add_argument('--debug', '-d', action='store_true', default=True, help='Enable debug mode')
    args = parser.parse_args()
    
    model, tokenizer = load(args.model)
    
    app.run(port=args.port, debug=args.debug)