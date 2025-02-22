import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from gpt4all import GPT4All
from htmlTemplates import css, bot_template, user_template
from typing import Optional, List
import os

# Configuration defaults (can be overridden by .env)
DEFAULT_MODEL_PATH = "C:/Users/atq765/GenAI/ask-multiple-pdfs-main/model/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Must be downloaded manually
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Wrapper class to adapt GPT4All to LangChain's LLM interface
class GPT4AllWrapper(LLM):
    def __init__(self, model_path: str, temperature: float = 0.7, **kwargs):
        super().__init__()
        self._model_path = model_path  # Changed to _model_path
        self._temperature = temperature  # Changed to _temperature
        self._model = GPT4All(model_path, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "gpt4all"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate(prompt, temp=self._temperature, max_tokens=1000)
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response[:response.index(stop_token)]
        return response

    @property
    def _identifying_params(self) -> dict:
        return {"model_path": self._model_path, "temperature": self._temperature}

def get_pdf_text(pdf_docs: List) -> str:
    """Extract text from uploaded PDFs."""
    if not pdf_docs:
        return ""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text

def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks for processing."""
    if not text:
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks: List[str]) -> Optional[FAISS]:
    """Create a FAISS vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks were generated!")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found! Please set it in .env.")
        return None
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversation_chain(vectorstore: Optional[FAISS]) -> Optional[ConversationalRetrievalChain]:
    """Initialize the conversation chain with a language model."""
    if vectorstore is None:
        st.error("Vectorstore is not initialized!")
        return None
    
    model_path = os.getenv("GPT4ALL_MODEL_PATH", DEFAULT_MODEL_PATH)
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Downloading or specify a valid path in .env.")
            # Optionally, prompt user to download: https://gpt4all.io/models
            return None
        llm = GPT4AllWrapper(model_path, temperature=0.7)
        st.info("Using GPT4All model.")
    except Exception as e:
        st.warning(f"GPT4All failed: {e}. Falling back to OpenAI.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("No OpenAI API key found!")
            return None
        llm = ChatOpenAI(api_key=api_key)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question: str) -> None:
    """Handle user input and display the conversation."""
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Please process some documents first!")
        return
    if not user_question.strip():
        st.warning("Please enter a question!")
        return
        
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if message.type == "human" else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    """Main function to run the Streamlit app."""
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                            st.warning("No text could be extracted from the PDFs!")
                            return
                            
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.warning("No text chunks were created!")
                            return
                            
                        vectorstore = get_vectorstore(text_chunks)
                        if vectorstore:
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            if st.session_state.conversation:
                                st.success("Processing completed!")
                            else:
                                st.error("Failed to initialize conversation chain!")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")

        if st.button("Reset"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.success("Conversation reset!")

if __name__ == '__main__':
    main()