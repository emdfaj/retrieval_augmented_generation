

import fitz  # PyMuPDF for PDF extraction
from docx import Document  # For extracting text from Word docs
from sentence_transformers import SentenceTransformer
import chromadb
import re

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ''
    for page in document:
        text += page.get_text()
    document.close()
    return text

# Function to extract text from Word documents
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to preprocess the extracted text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and multiple spaces
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to vectorize text chunks using SentenceTransformer
def vectorize_text(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

# Function to index text and embeddings into ChromaDB
def index_text(collection, text_chunks, embeddings):
    for i, chunk in enumerate(text_chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[{"id": str(i), "source": "document"}],
            ids=[str(i)]
        )

# Main script to extract, preprocess, and index data
if __name__ == "__main__":
    # Extract text from files
    pdf_path = 'resume.pdf'  # Replace with your file path
    extracted_text = extract_text_from_pdf(pdf_path)

    # Preprocess text
    preprocessed_text = preprocess_text(extracted_text)

    # Split preprocessed text into chunks (e.g., 200 characters per chunk)
    text_chunks = [preprocessed_text[i:i+200] for i in range(0, len(preprocessed_text), 200)]

    # Vectorize the text chunks
    embeddings = vectorize_text(text_chunks)

    # Initialize ChromaDB client and create collection
    client = chromadb.Client()
    collection = client.create_collection("personal_data", get_or_create=True)

    # Index the text chunks and embeddings into the collection
    index_text(collection, text_chunks, embeddings)

    print("Data extraction, preprocessing, and indexing completed successfully.")
