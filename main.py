import os
import pickle
import asyncio
import gc
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import time

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini API Key is missing. Please set it in the .env file.")

genai.configure(api_key=gemini_api_key)

# FastAPI App Initialization
app = FastAPI(title="Tripzoori AI Assistant API", description="API for answering questions about Tripzoori", version="1.0")

# Website URL
TRIPZOORI_URL = "https://tripzoori-gittest1.fly.dev/"
VECTOR_STORE_PATH = "vector_store.pkl"

# Global variable to store retriever
vector_store = None

# Function to extract text only from HTML content
def extract_text_from_html(html_content):
    """Extract meaningful text from HTML while removing scripts, styles, etc."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text(separator=' ', strip=True)
    
    # Remove extra whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

# Function to Load and Process Website Content with Playwright - Memory Optimized
async def load_website_content_async():
    try:
        # Set up Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set timeout to avoid hanging
            await page.goto(TRIPZOORI_URL, wait_until="networkidle", timeout=20000)
            html_content = await page.content()
            await browser.close()
        
        # Extract text to reduce memory usage
        text_content = extract_text_from_html(html_content)
        document = Document(page_content=text_content, metadata={"source": TRIPZOORI_URL})
        
        # Memory efficient text splitting - smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        document_chunks = text_splitter.split_documents([document])
        
        # Free memory
        del html_content, text_content, document
        gc.collect()
        
        # Use the smallest practical embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Improves quality with minimal overhead
        )
        
        # Create vector store with smaller dimensions
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        
        # Save vector store to disk
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(vector_store, f)
        
        # Free memory
        del document_chunks, embeddings
        gc.collect()
        
        return vector_store
    except Exception as e:
        raise Exception(f"Error fetching website content: {str(e)}")

# Function to Load Persisted Vector Store
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None
    return None

# Prompt Template - Simplified
prompt_template = """
You are Tripzoori AI Assistant, designed to answer questions about the website 'tripzoori-gittest1.fly.dev'.
Use the following context to provide accurate responses:

{context}

Question: {question}
"""

# Function to Generate Responses using Gemini - Memory Optimized
def generate_response(context, question):
    try:
        model = genai.GenerativeModel("gemini-1.0-pro")  # Using smaller model
        final_prompt = prompt_template.format(context=context, question=question)
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to clean up memory
def cleanup_memory():
    gc.collect()

# Request Model for API Input
class QueryRequest(BaseModel):
    question: str

# Initialize endpoint
@app.on_event("startup")
async def startup_event():
    global vector_store
    # Try to load the vector store at startup
    vector_store = load_vector_store()

# GET API Endpoint with memory management
@app.get("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori_get(
    background_tasks: BackgroundTasks,
    question: str = Query(..., description="Question about Tripzoori")
):
    global vector_store
    try:
        # Initialize vector store if needed
        if vector_store is None:
            vector_store = await load_website_content_async()
        
        # Reduced k to minimize memory usage
        retrieved_docs = vector_store.similarity_search(question, k=1)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        response = generate_response(context, question)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_memory)
        
        return {"answer": response}
    except Exception as e:
        # Clean up on error
        background_tasks.add_task(cleanup_memory)
        raise HTTPException(status_code=500, detail=str(e))

# POST API Endpoint with memory management
@app.post("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori(request: QueryRequest, background_tasks: BackgroundTasks):
    global vector_store
    try:
        # Initialize vector store if needed
        if vector_store is None:
            vector_store = await load_website_content_async()
        
        # Reduced k to minimize memory usage
        retrieved_docs = vector_store.similarity_search(request.question, k=1)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        response = generate_response(context, request.question)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_memory)
        
        return {"answer": response}
    except Exception as e:
        # Clean up on error
        background_tasks.add_task(cleanup_memory)
        raise HTTPException(status_code=500, detail=str(e))

# Simpler root endpoint
@app.get("/")
async def root():
    return {"message": "Tripzoori AI Assistant is running. Use /ask endpoint with a question parameter."}
