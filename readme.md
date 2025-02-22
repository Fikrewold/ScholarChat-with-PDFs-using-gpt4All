# MultiPDF Chat App

> You can find the tutorial for this project on [YouTube](https://youtu.be/dXxQ0LR-3Hg).

## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

# Project Overview

This application provides responses to your questions by leveraging a local language model and processing text content. Below is an outline of how it works:

## How It Works

The application follows these steps to generate responses:

### Example Code
# Project Overview

This application provides responses to your questions by leveraging a local language model and processing text content. Below is an outline of how it works:

## How It Works

The application follows these steps to generate responses:

### Example Code

```python
from gpt4all import GPT4All
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # Downloads/loads a 4.66GB LLM
with model.chat_session():
    print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))
```

 Troubleshooting

If the above code doesn’t work, manually download the model and save it in the model folder.
Processing Steps
Processing Steps

## How It Works

### PDF Loading
The app reads multiple PDF documents and extracts their text content.

### Text Chunking
The extracted text is split into smaller, manageable chunks for efficient processing.

### Language Model
A language model generates vector representations (embeddings) of the text chunks.

### Similarity Matching
When you ask a question, the app compares it to the text chunks and identifies the most semantically similar ones.

### Response Generation
The selected chunks are fed into the language model, which generates a response based on the relevant content from the PDFs.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:


   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Contributing
------------
This repository is intended for educational purposes and does not accept further contributions. It serves as supporting material for a YouTube tutorial that demonstrates how to build this project. Feel free to utilize and enhance the app based on your own requirements.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).
