"""
    Objetivo: Criar um agente com memória de curto prazo que utilize a busca por similaridade em um vector store a fim de responder perguntas.

    Passos
    1. pip install -r requirements.txt
    2. ollama serve
    3. ollama pull llama3.1:8b
    4. ollama pull nomic-embed-text
    5. python example3.py
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig
from rich import print
from rich.markdown import Markdown

# 1. Inicializar modelo llm
model = init_chat_model(model="ollama:llama3.1:8b")

# 2. Inicializar modelo de embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Inicializar vector store
vector_store = InMemoryVectorStore(embedding=embeddings)

# 4. Carregar documentos
file_path = "./3DTAlpha.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# 5. Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
ids = vector_store.add_documents(documents=all_splits)

# 6. Define tool de busca por similaridade
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Use esta ferramenta apenas quando precisar consultar informações acerca do jogo RPG 3D&T a fim de auxiliar na respota para o usuário
    
    Args:
        query: string que representa os termos a serem buscados num vector store através da pesquisa por similaridade

    Returns:
        Retorna os documentos que mais se assemelham de acordo com os termos buscados
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

# 7. Cria agente definindo modelo, ferramentas e memória de curto prazo
checkpointer = InMemorySaver()
config = RunnableConfig(configurable={"thread_id": 1})
agente = create_agent(model=model, tools=tools, checkpointer=checkpointer)

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        print(Markdown("---"))

        if user_input.lower() in ["/quit", "/exit"]:
            print("Até logo 👋")
            print(Markdown("---"))
            break

        system_message = SystemMessage(
            "Você é um assistente útil e amigável focado em responder as perguntas do usuário."
            "Você tem acesso a uma ferramenta com as regras do jogo de RPG 3D&T. Use a ferramenta apenas quando o usuário solicitar informações sobre 3D&T."
            "Se você não souber responder a pergunta do usuário, diga que não sabe."
            "Se comunique usando o português brasileiro."
        )
        human_message = HumanMessage(user_input)
        response = agente.invoke({"messages": [system_message, human_message]}, config=config)

        print(response["messages"][-1].content)
        print(Markdown("---"))