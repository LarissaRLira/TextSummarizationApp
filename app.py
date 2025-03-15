#IMPORTANDO BIBLIOTECAS

from langchain_anthropic import ChatAnthropic
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter  
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv, find_dotenv
import os

# CARREGANDO AS VARIÁVEIS DE AMBIENTE
load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# CRIAR MODELO AI
llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature= 0, # Ajusta o nível de criatividade do modelo
    anthropic_api_key = ANTHROPIC_API_KEY
)

text = "Caso precise colocar no PATH vai em Pesquisar -> variaveis de ambiente -> Editar variaveis de ambiente -> Ai vai abrir uma aba e vc clica em variaveis de ambiente"

#SPLIT TEXT: Fatiamento do texto 
text_splitter = CharacterTextSplitter()
texts = text_splitter.split_text(text)

#CREATE DOCUMENTS:
docs = [Document(page_content=text) for text in texts]

#SUMMARIZAÇÃO DOS TEXTOS
chain = load_summarize_chain(llm =llm, chain_type="stuff")

#EXECUTAR A CHAIN 
summary=chain.invoke(docs)
    
print(summary['output_text'])
