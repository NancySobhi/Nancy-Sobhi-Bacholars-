from langchain_ollama import OllamaLLM

# This tells the computer to use the "Brain" we just downloaded
llm = OllamaLLM(model="llama3.2")

# Let's ask it a simple question
response = llm.invoke("Hello, are you working?")

# This prints the answer in your terminal
print(response)