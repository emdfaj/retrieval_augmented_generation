

import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Initialize ChromaDB client and load the collection
client = chromadb.Client()
collection = client.create_collection("personal_data", get_or_create=True)

# Initialize the SentenceTransformer model for embedding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve relevant data from ChromaDB
def retrieve_relevant_data(question, collection, model):
    question_embedding = model.encode([question])[0]
    results = collection.query(query_embeddings=[question_embedding], n_results=5)
    return results['documents'][0] if results['documents'] else "No relevant data found."

# Function to handle Gradio input and output
def answer_question(question):
    response = retrieve_relevant_data(question, collection, model)
    return response

# Set up Gradio interface
gr_interface = gr.Interface(
    fn=answer_question,  # The function that processes user input
    inputs="text",       # Input type is a text box
    outputs="text",      # Output is a text box showing the AI's response
    title="Conversational AI with RAG",
    description="Ask me anything about the indexed data!"
)

# Launch Gradio interface
if __name__ == "__main__":
    gr_interface.launch(share=True)
