import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from gpt4all import GPT4All
import numpy as np

# Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    # Using Hugging Face's pre-trained embeddings model (free option)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain using GPT4All for answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    # Load GPT4All model (you can change to another model or fine-tune it)
    model = GPT4All("ggml-gpt4all-j-v1.3-groovy.bin")  # You can replace with other available models

    # Create a prompt using the template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the pre-saved FAISS vector store for searching
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Generate a response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Output the response
    st.write("Reply: ", response["output_text"])

# Streamlit app UI
st.title("PDF QA App with GPT4All")
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    # Get the text from the uploaded PDFs
    text = get_pdf_text(uploaded_files)
    
    # Get text chunks
    text_chunks = get_text_chunks(text)
    
    # Create a vector store from the chunks
    get_vector_store(text_chunks)

    # Ask the user for a question
    user_question = st.text_input("Ask a question about the content of the PDFs:")

    if user_question:
        user_input(user_question)
