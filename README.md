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
```

2. Navigate to the project directory:
```bash
cd your-repo-name
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## OpenAI API Key
You will need an OpenAI API key to use this program. Follow these steps to set it up:

1. Check the .env.sample file in the repository to see the required environment variables.

2. Create a .env file in the project root (same directory as main_1.py) and add your OpenAI API key in the following format:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage
1. Place the Documents:

Ensure that the resume PDF file or any other documents you want to process are placed in the same directory as the main_1.py file. The program will use these files for analysis or processing.

2. Run the Program:

To execute the program, use the following command:
```bash
python main_1.py
```

This will process the documents in the directory and output the results based on the implemented logic.
