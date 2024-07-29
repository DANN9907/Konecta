
# Langchain Chatbot for Bruno_child_offers PDFs

This chatbot is crafted to respond to inquiries regarding the Bruno_child_offers PDF, utilizing OpenAI LLM models.

## Objective

The aim of this project is to create a chatbot that can answer questions about any uploaded PDF, specifically the 'Bruno_child_offers' in this instance. The chatbot leverages natural language processing and machine learning methods to comprehend user questions and fetch pertinent information from the PDFs.

## Key Features

- Single PDF Upload: Allows for the upload of one PDF document at a time.
- Conversational Retrieval: Employs conversational retrieval methods to deliver contextually relevant responses to user questions.
- Language Models: Integrates Hugging Face models for natural language comprehension and generation, facilitating meaningful interactions with the chatbot.
- PDF Text Extraction: Processes PDF files to extract their textual content, which is then used for indexing and retrieval purposes.
- Text Chunking: Divides the extracted text into smaller segments to enhance retrieval efficiency and provide more accurate answers.

## Installation

To set up and run the Chatbot, follow these steps:

1. Clone the repository:
   ```bash
   git clone
   ```

2. Install all dependencies and create a virtual environment:
   ```bash
   pip install virtualenv
   python<version> -m venv <virtual-environment-name>
   <virtual-environment-name>/Scripts/activate
   ```

3. Add your Hugging Face API key by creating a .env file in the project folder and include the following line:
   ```
   HUGGINGFACEHUB_API_TOKEN="Your key"
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```
