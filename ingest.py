import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

def ingest_data():
    pdf_path = os.path.join("data", "Document.pdf")
    persist_directory = "chroma_db"
    
    print(f"Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    import time
    
    print(f"Creating embeddings and storing in ChromaDB at {persist_directory}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    batch_size = 10
    print(f"Total chunks: {len(splits)}. Ingesting in batches of {batch_size} to avoid rate limits...")
    print("Waiting 15 seconds for API quota to reset...")
    time.sleep(15)
    
    # Initialize empty vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1} of {(len(splits)-1)//batch_size + 1} ({len(batch)} chunks)...")
        vectorstore.add_documents(documents=batch)
        if i + batch_size < len(splits):
            print(f"Sleeping for 10 seconds to avoid API rate limits...")
            time.sleep(10)
    

    
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
