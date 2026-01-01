"""
    Objetivo: Realizar busca por similaridade de texto em um documento PDF.

    Passos
    1. pip install langchain-core langchain-community langchain-ollama langchain-text-splitters pypdf
    2. ollama serve
    3. ollama pull nomic-embed-text
    4. python rag2.py
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings

# Inicializar o modelo de embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Carregar o arquivo PDF
file_path = "./3DTAlpha.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Dividir o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search_with_score("Como funciona a vantagem Aceleração?")
doc1, score1 = results[0]
doc2, score2 = results[1]

print(f"Quantidade de resultados: {len(results)}")
print(f"Score do primeiro resultado: {score1}")
print(f"Documento do primeiro resultado: {doc1.page_content}")

print(f"Score do segundo resultado: {score2}")
print(f"Documento do segundo resultado: {doc2.page_content}")