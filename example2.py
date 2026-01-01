"""
    Objetivo: Criar um agente que utilize a busca por similaridade em um vector store a fim de responder uma pergunta.

    Passos
    1. pip install langchain langchain-core langchain-community langchain-ollama langchain-text-splitters pypdf
    2. ollama serve
    3. ollama pull llama3.1:8b
    4. ollama pull nomic-embed-text
    5. python example2.py
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent

# 1. Inicializar o modelo LLM
model = ChatOllama(model="llama3.1:8b")

# 2. Inicializar o modelo de embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Inicializar o vector store
vector_store = InMemoryVectorStore(embeddings)

# 4. Carregar o arquivo PDF
file_path = "./3DTAlpha.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# 5. Dividir o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = (
    "Você tem acesso a uma ferramenta que recupera informações de um documento PDF."
    "Use a ferramenta para ajudar a responder as perguntas do usuário."
    "Se você não souber a resposta, diga que não sabe."
    "Se comunique usando o português brasileiro."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "Por qual lei de direitos autorais o produto 3DTAlpha é protegido?\n\n"
    "Quando descobrir me diga também a data de publicação do produto."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()