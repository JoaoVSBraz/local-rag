# Retrieval-Augmented Generation

A finalidade do presente repositório é criar um chat que recupere informações de uma base de conhecimentos a fim de embasar a resposta do modelo a uma pergunta.

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- [Ollama](https://ollama.com/) instalado e em execução

1. Clone o repositório: git clone <url-do-repositorio>
2. Crie um ambiente virtual (recomendado): python -m venv .venv

## Plano de Ação

1. Compreender os aspectos fundamentais da linguagem python e técnica RAG através da ferramenta LangChain.
2. Realizar busca semântica através da busca por similaridade numa base de conhecimentos.
3. Criar um chat que utilize a busca semântica para contextualizar e embasar resposta a uma pergunta.
4. Melhorar o chat a fim de que possa se lembrar de conversas anteriores.
5. Utilizar o pgvector para busca semântica na base de conhecimentos.
6. Fornecer interface de usuário para comunicação com o modelo desenvolvido.

## Referências

- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.ollama.com/