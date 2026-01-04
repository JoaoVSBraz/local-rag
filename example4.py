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
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig
from langgraph.graph.message import MessagesState, add_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import Annotated, TypedDict
from typing import Literal
from rich import print
from rich.markdown import Markdown

# 1. Inicializar modelo llm
model = init_chat_model(model="ollama:llama3.1:8b", temperature=0.3)

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
@tool
def rag(query: str):
    """IMPORTANTE: Use esta ferramenta APENAS e EXCLUSIVAMENTE quando o usuário perguntar especificamente sobre:
    - O jogo de RPG 3D&T (Terra de Dentro e Terra de Fora)
    - Regras, mecânicas, personagens, habilidades, vantagens, desvantagens do 3D&T
    - Qualquer coisa relacionada ao sistema de RPG 3D&T
    
    NÃO use esta ferramenta para:
    - Perguntas gerais sobre RPGs em geral
    - Perguntas sobre outros jogos
    - Perguntas sobre qualquer outro assunto que não seja especificamente o jogo 3D&T
    
    Args:
        query: termos de busca relacionados especificamente ao jogo 3D&T

    Returns:
        Documentos do manual do 3D&T que correspondem à busca
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

tools = [rag]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 8. Cria state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int

# 9. Define node que chama o agente
def llm_call(state: dict):
    """LLM decide quando chamar ou não a tool"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        """
                        Você é um agente útil e amigável encarregado de responder perguntas do usuário.

                        REGRAS CRÍTICAS SOBRE O USO DA FERRAMENTA 'rag':
                        
                        1. Use a ferramenta 'rag' APENAS quando o usuário perguntar ESPECIFICAMENTE sobre:
                           - O jogo de RPG 3D&T
                           - Regras, mecânicas, personagens, habilidades, vantagens, desvantagens do 3D&T
                           - Qualquer coisa relacionada ao sistema de RPG 3D&T
                        
                        2. NÃO use a ferramenta 'rag' para:
                           - Perguntas gerais sobre RPGs (ex: "O que é um RPG?", "Como jogar RPG?")
                           - Perguntas sobre outros jogos (D&D, Pathfinder, etc.)
                           - Perguntas sobre qualquer assunto que NÃO seja especificamente o jogo 3D&T
                           - Perguntas sobre programação, ciência, história, etc.
                        
                        3. Se a pergunta NÃO for especificamente sobre 3D&T, responda diretamente SEM usar a ferramenta.
                        
                        4. Se você não souber a resposta, diga que não sabe. NÃO invente informações.
                        
                        5. Se comunique utilizando o Português Brasileiro.
                        
                        6. Quando usar a ferramenta, responda de forma clara e concisa baseada nos resultados, sem mencionar que usou uma ferramenta.
                        """
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 10. Define node que executa a tool e retorna o resultado
def tool_node(state: dict):
    """Executa a chamada da tool"""
    
    response = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        response.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": response}

# 11. Define node condicional que decide o próximo passo do fluxo de execução no grafo
def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Decide se devemos continuar o loop baseado se a LLM fez uma tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    print(last_message)

    if last_message.tool_calls:
        return "tool_node"

    return END

# 12. Cria fluxo de build
agent_builder = StateGraph(AgentState)

# 13. Adiciona nodes e edges
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]    
)
agent_builder.add_edge("tool_node", "llm_call")

# 14. Compila o agente
checkpointer = InMemorySaver()
config = RunnableConfig(configurable={"thread_id": 1})
agente = agent_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        print(Markdown("---"))

        if user_input.lower() in ["/quit", "/exit"]:
            print("Até logo 👋")
            print(Markdown("---"))
            break
       
        messages = [HumanMessage(user_input)]
        response = agente.invoke({"messages": messages, "llm_calls": 0}, config)

        print(response["messages"][-1].content)
        print(Markdown("---"))