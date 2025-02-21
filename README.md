# ScholarChat with PDFs using GPT4All

## Overview
ScholarChat with PDFs is a Streamlit-based application that allows users to upload PDF documents and interact with their content using an AI-powered chatbot. The app leverages GPT4All for generating responses and FAISS for efficient document search and retrieval.
![image](https://github.com/user-attachments/assets/bfa7a5e2-d156-48e5-92f8-4c6b698cba44)
## Features
- **Upload multiple PDFs** and extract text from them.
- **Chunk the extracted text** into manageable pieces for better processing.
- **Create a vector store** using FAISS and pre-trained Hugging Face embeddings.
- **Use GPT4All** to provide intelligent answers based on uploaded document content.
- **Streamlit UI** for a seamless user experience.

## Technologies Used
- **Streamlit**: For building the interactive web interface.
- **PyPDF2**: For extracting text from PDF documents.
- **LangChain**: For text processing and integration with AI models.
- **FAISS**: For efficient vector search and similarity matching.
- **GPT4All**: For answering user queries based on PDF content.
- **Hugging Face Embeddings**: For generating vector representations of text.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/ScholarChat-with-PDFs-using-GPT4All.git
cd ScholarChat-with-PDFs-using-GPT4All

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Running the App
Once dependencies are installed, launch the Streamlit app:

```bash
streamlit run app.py
```

## How It Works
1. Upload PDF files through the Streamlit UI.
2. The application extracts text from the PDFs.
3. The extracted text is split into chunks for better processing.
4. FAISS creates an efficient vector store from the chunks.
5. The user can ask questions about the PDF content.
6. The app retrieves relevant chunks and generates an AI-powered response using GPT4All.

## File Structure
```
ScholarChat-with-PDFs-using-GPT4All/
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
```

## Dependencies
The required Python packages are listed in `requirements.txt`:
```
streamlit
google-generativeai
python-dotenv
langchain
PyPDF2
chromadb
faiss-cpu
langchain_google_genai
```

## Future Improvements
- Support for additional document formats (DOCX, TXT, etc.).
- Integration with more powerful LLMs.
- Improved UI/UX with enhanced visualization.
- Deployment as a web service.

## License
This project is licensed under the MIT License.

## Author
Developed by **Fikrewold Bitew**

