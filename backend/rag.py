import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# Wrapper for Google Gemini Embeddings to work with ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(input)

# Global collection variable
collection = None

def add_document_to_db(text: str, source: str, doc_id: str):
    """Adds a document to the vector database."""
    if collection:
        collection.add(
            documents=[text],
            metadatas=[{"source": source}],
            ids=[doc_id]
        )

def ingest_pdf(file_path: str, source_name: str):
    """Reads a PDF and adds its content to the vector DB."""
    if not collection:
        print("Collection not initialized.")
        return

    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                add_document_to_db(text, f"{source_name} (Page {i+1})", f"{source_name}_p{i+1}")
        print(f"Successfully ingested {source_name}")
    except Exception as e:
        print(f"Error ingesting PDF {file_path}: {e}")

def query_medical_db(query: str, n_results: int = 2) -> str:
    """Queries the local vector database for relevant medical info."""
    if not collection:
        return "Trusted Medical Library is currently unavailable."

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return "No relevant documents found in the local trusted library."
            
        context = ""
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            context += f"Source: {source}\nContent: {doc}\n\n"
            
        return context
    except Exception as e:
        return f"Error querying vector DB: {e}"

# Initialize RAG
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    gemini_ef = GeminiEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(name="medical_knowledge", embedding_function=gemini_ef)
    
    # Seed data if empty
    if collection.count() == 0:
        print("Seeding Vector DB with initial data...")
        initial_data = [
            ("Vaccines are safe and effective. They do not cause autism. Multiple studies have debunked this myth.", "CDC - Vaccine Safety"),
            ("Drinking bleach is extremely dangerous and can be fatal. It does not cure COVID-19 or any other virus.", "WHO - Mythbusters"),
            ("5G networks use non-ionizing radio waves and do not spread viruses like COVID-19.", "WHO - 5G and Health"),
            ("There is no scientific evidence that an alkaline diet prevents cancer. The body maintains its own pH balance.", "American Institute for Cancer Research"),
            ("Lemons contain Vitamin C but do not cure cancer. Cancer requires professional medical treatment.", "National Cancer Institute"),
            ("Ivermectin is an anti-parasitic drug and is not approved for treating viral infections like COVID-19.", "FDA"),
            ("Masks are effective at reducing the spread of respiratory droplets and viruses.", "CDC - Mask Guidance"),
            ("Antibiotics only kill bacteria, not viruses. They should not be used for colds or flu.", "CDC - Antibiotic Use"),
            ("Sugar intake should be limited, but it does not directly cause hyperactivity in children.", "WebMD"),
            ("Detox teas and diets are generally unnecessary as the liver and kidneys detoxify the body naturally.", "Mayo Clinic")
        ]
        for i, (text, source) in enumerate(initial_data):
            add_document_to_db(text, source, f"seed_{i}")

    rag_tool = Tool(
        name="Trusted Medical Library (RAG)",
        func=query_medical_db,
        description="ALWAYS use this first. Searches a local database of verified medical facts from CDC, WHO, and other trusted sources."
    )

except Exception as e:
    print(f"RAG Initialization Error: {e}")
    def rag_fallback(query: str):
        return "Trusted Medical Library is currently unavailable. Please use web search."
        
    rag_tool = Tool(
        name="Trusted Medical Library (RAG)",
        func=rag_fallback,
        description="Unavailable."
    )
