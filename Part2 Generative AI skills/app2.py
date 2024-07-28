import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css

from transformers import pipeline

def get_pdf_text(pdf_files):
    """
    Extracts text from a list of PDF files.

    Args:
        pdf_files (list): List of uploaded PDF files.

    Returns:
        str: The extracted text from all the PDF files combined.
    """
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    """
    Splits the extracted text into smaller chunks.

    Args:
        text (str): The text to be split into chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Generates a vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        FAISS: A vector store created from the text chunks.
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    """
    Sets up a conversational retrieval chain using a vector store.

    Args:
        vector_store (FAISS): The vector store to be used for retrieval.

    Returns:
        ConversationalRetrievalChain: A chain for handling conversations based on the vector store.
    """
    llm = pipeline("text2text-generation", model="google/flan-t5-xxl")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question):
    """
    Handles user input and updates the chat history.

    Args:
        question (str): The user's question.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

def main():
    """
    The main function to run the Streamlit app for PDF Q&A.

    Loads environment variables, sets up the page, and handles user interactions for uploading PDFs and asking questions.

    Returns:
        None
    """
    load_dotenv()
    st.set_page_config(page_title="PDF Q&A")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Q&A")

    user_question = st.text_input("Make a question:")
    if user_question:
        handle_user_input(user_question)
        st.text_input("Make another question:", key="next_question")

    pdf_files = st.file_uploader("Upload the Bruno_child_offers PDF", type=["pdf"], accept_multiple_files=False)
    if pdf_files:
        with st.spinner("Extracting text from PDFs..."):
            raw_text = get_pdf_text(pdf_files)

        with st.spinner("Splitting text into fragments..."):
            text_chunks = get_chunk_text(raw_text)

        with st.spinner("Generating vector store..."):
            vector_store = get_vector_store(text_chunks)

        with st.spinner("Setting up conversation thread..."):
            st.session_state.conversation = get_conversation_chain(vector_store)

        st.success("Done! You can now ask questions about your PDFs.")

if __name__ == "__main__":
    main()