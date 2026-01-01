"""
    Objetivo: Conversar com um modelo de IA da forma mais simples possível.
    Passos:
    1. pip install langchain langchain-ollama
    2. ollama serve
    3. ollama pull llama3.1:8b
    4. python example0.py
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

# Inicializar o modelo de IA usando init_chat_model
# Formato: "ollama:modelo" ou apenas "modelo" (com inferência automática)
model = init_chat_model(model="ollama:llama3.1:8b")

system_message = SystemMessage(content="Você é um assistente de IA que responde perguntas de forma amigável e informativa.")
messages = [system_message]

response = model.invoke(messages)
while True:
    print(f"{'Human':-^80}")
    user_input = input("Você: ")
    human_message = HumanMessage(content=user_input)

    if(user_input.lower() == "sair" or user_input.lower() == "exit"):
        break

    messages.append(human_message)
    response = model.invoke(messages)
    print(f"{'Assistant':-^80}")
    print(response.content)
    messages.append(response)
