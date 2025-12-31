import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração de Logs (opcional, para ver o que acontece por baixo dos panos)
logging.basicConfig(level=logging.INFO)

# --- CONFIGURAÇÕES ---
# String de conexão com o Banco (ajuste a senha conforme seu setup)
CONNECTION_STRING = "postgresql+psycopg://postgres:admin@localhost:5433/postgres"
COLLECTION_NAME = "documentos_empresa"


# 1. Inicializar os Modelos do Ollama
print("--- Inicializando Modelos Ollama ---")
# Modelo de Embeddings (transforma texto em números)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Modelo LLM (gera as respostas)
llm = ChatOllama(model="llama3")

# 2. Configurar o PGVector (Vector Store)
print("--- Conectando ao PGVector ---")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# 3. Ingestão de Dados (Exemplo)
# Em um caso real, você carregaria PDFs ou TXTs aqui.
# docs = [
#     Document(
#         page_content="A política de home office da empresa permite 3 dias de trabalho remoto por semana.",
#         metadata={"source": "rh_manual_v2.pdf", "topico": "beneficios"}
#     ),
#     Document(
#         page_content="O reembolso de despesas de viagem deve ser solicitado até o dia 5 do mês seguinte.",
#         metadata={"source": "financeiro_policy.txt", "topico": "financeiro"}
#     ),
#     Document(
#         page_content="O projeto Alpha visa migrar toda a infraestrutura para Kubernetes até o fim do ano.",
#         metadata={"source": "tech_roadmap.md", "topico": "engenharia"}
#     )
# ]

# --- EXEMPLO: Carregando de um PDF (Descomente para usar) ---
# 1. Instale: pip install pypdf
pdf_path = "./3DTAlpha.pdf"
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load()
# 
# # É importante dividir PDFs grandes em pedaços menores (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pdf_docs)
# 
# # Substitua 'docs' por 'splits' na linha abaixo se for usar o PDF
docs = splits
# ------------------------------------------------------------

print("--- Adicionando documentos ao Banco Vetorial ---")
# Adiciona os documentos. O PGVector converte para vetores automaticamente aqui.
vector_store.add_documents(docs)
print(f"Inseridos {len(docs)} documentos com sucesso.")

# 4. Configurar o Sistema RAG
# Transformamos o vector store em um "retriever" (buscador)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2} # Retorna os 2 trechos mais relevantes
)

# Template do Prompt que será enviado ao LLM
template = """
Você é um assistente útil. Use APENAS o contexto fornecido abaixo para responder à pergunta.
Se você não souber a resposta baseada no contexto, diga que não sabe.

Contexto:
{context}

Pergunta:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Função auxiliar para formatar os documentos recuperados em uma string única
def format_docs(docs):
    formatted_content = "\n\n".join([d.page_content for d in docs])
    print("\n--- CONTEXTO RECUPERADO (DEBUG) ---")
    print(formatted_content)
    print("-----------------------------------")
    return formatted_content

# A "Chain" (Corrente) de execução
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execução (Chat com seus dados)
pergunta = "Levando em conta seus conhecimentos sobre o 3DTAlpha, acesse a explicação sobre Vantagens na parte 3 e me diga qual o custo da vantamgem Arcano e sua descrição."

print(f"\n--- Pergunta: {pergunta} ---")
print("Gerando resposta...")

response = rag_chain.invoke(pergunta)

print("\n--- Resposta do LLM: ---")
print(response)

# Limpeza (Opcional): Se quiser limpar a tabela depois do teste
# vector_store.delete_collection()