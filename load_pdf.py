from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("financial_data.pdf")
docs = loader.load()

# 2. Split it into bite-sized pieces
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Save into a local database
print("Saving to database (this will download a tool first, please wait)...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OllamaEmbeddings(model="nomic-embed-text"), 
    persist_directory="./chroma_db"
)

print("Success! Your PDF is now processed and ready.")