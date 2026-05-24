from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM

# 1. Load the PDF
loader = PyPDFLoader("financial_data.pdf")
pages = loader.load()

# 2. Combine all text into one big block
full_text = "\n".join([page.page_content for page in pages])

# 3. Initialize the AI
llm = OllamaLLM(model="llama3.2")

print("Chatbot is ready! (Type 'quit' to exit)")

while True:
    query = input("Ask a question about your PDF: ")
    if query.lower() == "quit":
        break
    
    # 4. "Stuff" the PDF text and the question into the AI
    # We limit text to avoid overwhelming the AI's memory
    prompt = f"Use this text to answer the question: {full_text[:15000]} \n\n Question: {query}"
    
    response = llm.invoke(prompt)
    print("\nAnswer: " + response + "\n")