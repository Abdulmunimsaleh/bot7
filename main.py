import os
import pickle
import asyncio
import gc
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from playwright.async_api import async_playwright
import google.generativeai as genai
from dotenv import load_dotenv
import psutil  # For monitoring memory usage

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini API Key is missing. Please set it in the .env file.")

genai.configure(api_key=gemini_api_key)

# FastAPI App Initialization
app = FastAPI(title="Tripzoori AI Assistant API", description="API for answering questions about Tripzoori", version="1.0")

# Website URL and Vector Store Path
TRIPZOORI_URL = "https://tripzoori-gittest1.fly.dev/"
VECTOR_STORE_PATH = "vector_store.pkl"

# Global cache for vector store
vector_store_cache = None

# Function to Load Persisted Vector Store
def load_vector_store():
    global vector_store_cache
    if vector_store_cache is not None:
        return vector_store_cache

    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                vector_store_cache = pickle.load(f)
                return vector_store_cache
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
    return None

# Function to Load and Process Website Content with Playwright
async def load_website_content_async(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            content = await page.content()
            document = Document(page_content=content, metadata={"source": url})

            # Memory-efficient text splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
            document_chunks = text_splitter.split_documents([document])

            # Use a lightweight embedding model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(document_chunks[:25], embeddings)  # Process in batches

            for i in range(25, len(document_chunks), 25):
                vector_store.add_documents(document_chunks[i:i + 25])

            # Save vector store to disk
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vector_store, f)

            return vector_store
        finally:
            await page.close()
            await browser.close()

# Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are Tripzoori AI Assistant, designed to answer questions about the website 'tripzoori-gittest1.fly.dev'.
    Use the following context to provide accurate responses:

    {context}

    Question: {input}
    """
)

# Function to Generate Responses using Gemini
MAX_CONTEXT_LENGTH = 1000  # Truncate context to avoid large prompts

def generate_response(final_prompt):
    try:
        truncated_prompt = final_prompt[:MAX_CONTEXT_LENGTH]
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(truncated_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Request Model for API Input
class QueryRequest(BaseModel):
    question: str

# Cleanup Resources
def cleanup_resources(*args):
    for obj in args:
        del obj
    gc.collect()

# Monitor Memory Usage
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory Usage: {memory_mb:.2f} MB")

    if memory_mb > 450:  # Warn if memory usage approaches the limit
        print("WARNING: Memory usage is approaching the 512MB limit!")

# GET API Endpoint
@app.get("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori_get(question: str = Query(..., description="Question about Tripzoori")):
    log_memory_usage()  # Log memory before processing

    try:
        vector_store = load_vector_store()
        if not vector_store:
            vector_store = await load_website_content_async(TRIPZOORI_URL)

        retrieved_docs = vector_store.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt.format(context=context, input=question)
        response = generate_response(final_prompt)

        cleanup_resources(retrieved_docs, context)  # Cleanup after processing
        log_memory_usage()  # Log memory after processing

        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST API Endpoint
@app.post("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori(request: QueryRequest):
    log_memory_usage()  # Log memory before processing

    try:
        vector_store = load_vector_store()
        if not vector_store:
            vector_store = await load_website_content_async(TRIPZOORI_URL)

        retrieved_docs = vector_store.similarity_search(request.question, k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt.format(context=context, input=request.question)
        response = generate_response(final_prompt)

        cleanup_resources(retrieved_docs, context)  # Cleanup after processing
        log_memory_usage()  # Log memory after processing

        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Tripzoori AI Assistant API. Use /ask endpoint to ask questions."}

# Startup Event to Preload Resources
@app.on_event("startup")
async def startup_event():
    global vector_store_cache
    vector_store_cache = load_vector_store()
    if not vector_store_cache:
        vector_store_cache = await load_website_content_async(TRIPZOORI_URL)