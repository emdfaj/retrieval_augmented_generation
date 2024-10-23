# LangChain-Based Retrieval-Augmented Generation (RAG) System
This project demonstrates the integration of data extraction, vectorization, and retrieval-augmented generation (RAG) using LangChain and OpenAI's GPT-3.5-turbo model. The system processes documents, indexes them in ChromaDB, and provides conversational AI capabilities with context-based question answering.

### Features
- **Data Extraction**: Supports text extraction from PDFs, Word documents, plain text files, and websites using Python libraries like PyMuPDF, python-docx, and BeautifulSoup.
- **Text Preprocessing**: Cleans and preprocesses extracted data by removing unnecessary characters and line breaks.
- **Text Vectorization**: Uses the SentenceTransformer model to vectorize text into embeddings for efficient retrieval.
- **Data Indexing**: Stores text embeddings in ChromaDB, making them retrievable for further question-answering tasks.
- **Question Embedding and Retrieval**: Embeds user queries and retrieves relevant information from ChromaDB using similarity-based search.
- **Conversational Retrieval with Memory**: Enables a conversational AI system that handles follow-up questions with context, using conversation history stored in memory.
- **Gradio Interface**: Provides an interactive web interface using Gradio, allowing users to ask questions and receive AI-generated answers based on the indexed data.

## Prerequisites
Before running this project, make sure you have the following:
- Python 3.x installed
- An OpenAI API key

### Installation
1. Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/your-repo-name.git

2. Navigate to the project directory:
```bash
cd your-repo-name

3. Install the required dependencies:
```bash
pip install -r requirements.txt
