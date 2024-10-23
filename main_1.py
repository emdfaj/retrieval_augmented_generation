

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Open the provided PDF file
    document = fitz.open(pdf_path)
    text = ''
    
    # Iterate over each page in the PDF
    for page in document:
        # Extract text from the page and add it to the overall text
        text += page.get_text()

    document.close()
    return text

# Specify the path to your PDF file
pdf_path = 'resume.pdf'
extracted_text = extract_text_from_pdf(pdf_path)
#print(extracted_text)

from docx import Document

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Example usage
#docx_text = extract_text_from_docx("document.docx")
#print(docx_text)

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example usage
#txt_text = extract_text_from_txt("notes.txt")
#print(txt_text)

import requests
from bs4 import BeautifulSoup

def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

# Example usage
web_text = extract_text_from_website("https://medium.com/@spaw.co/best-websites-to-practice-web-scraping-9df5d4df4d1")
#print(web_text)

import re

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and multiple spaces
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    return text.strip()

# Use the previously extracted text
preprocessed_text = preprocess_text(extracted_text)
#print(preprocessed_text)

# Vectorizing Text Chunks with Sentence Transformers
from sentence_transformers import SentenceTransformer

def vectorize_text(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

# Assuming text is chunked already
text_chunks = [preprocessed_text[i:i+200] for i in range(0, len(preprocessed_text), 200)]
embeddings_1 = vectorize_text(text_chunks)

#Indexing Data into ChromaDB

import chromadb

client = chromadb.Client()
collection = client.create_collection("personal_data", get_or_create=True)

def index_text(collection, text_chunks, embeddings):
    for i, chunk in enumerate(text_chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[{"id": str(i), "source": "resume"}],
            ids=[str(i)]
        )

index_text(collection, text_chunks, embeddings_1)

# Question embedding and retrieval from ChromaDB
# Define the model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model

def retrieve_relevant_data(question, collection, model):
    # Vectorize the input question
    question_embedding = model.encode([question])[0]
    # Retrieve the closest match from ChromaDB
    results = collection.query(query_embeddings=[question_embedding], n_results=5)
    return results['documents'][0]


# Setting up ChromaDB as a Retriever with Hugging Face embeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB vector store with collection name and embeddings
vector_store = Chroma(
    collection_name="personal_data",
    embedding_function=embeddings
)

# Convert ChromaDB collection to a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initializing OpenAI's GPT 3.5-Turbo for Language Model Integration

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key not found in enviroment variables")

print(openai_api_key)

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI API
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Setting up a Retrieval-Augmented Generation (RAG) Chain for answering question

from langchain.chains import RetrievalQA

# Set up the retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Optional: to return the documents retrieved from Chroma
)

# Ask a question using the chain
question = "What is my experience in Data Science?"
result = qa_chain({"query": question})

# Print the generated answer and the source documents
print("Generated Answer:", result['result'])
#print("Retrieved Documents:", result['source_documents'])


# Creating a Conversational Retrieval Chain with Memory for Follow-up questions
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


# Initialize memory to store conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Use the from_llm method to create a conversational retrieval chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,  # ChromaDB retriever
    memory=memory
)

# Ask the first question
question1 = "What is my experience in web development?"
response1 = conversational_chain.run(question1)
print(f"Q: {question1}\nA: {response1}")

# Follow-up question
question2 = "Can you explain my most recent project?"
response2 = conversational_chain.run(question2)
print(f"Q: {question2}\nA: {response2}")

# Setting up a Gradio Interface for Conversational AI with RAG
import gradio as gr

# Define a function to process user input and return AI response
def answer_question(question):
    response = conversational_chain.run(question)
    return response

# Set up the Gradio interface
gr_interface = gr.Interface(
    fn=answer_question,  # The function that processes user input
    inputs="text",  # Input type is a simple text box
    outputs="text",  # Output is a text box showing the AI's response
    title="Conversational AI with RAG",
    description="Ask me anything about my experience!"
)

# Launch the interface
gr_interface.launch()