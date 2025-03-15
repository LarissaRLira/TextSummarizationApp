# Text Summarization App

## Descrição

Este projeto utiliza a API da Anthropic com o LangChain para criar um modelo de Inteligência Artificial capaz de resumir textos automaticamente. O código carrega um texto, divide-o em partes menores, cria documentos a partir dessas partes e, finalmente, gera um resumo utilizando um modelo da Anthropic.

## Tecnologias Utilizadas

- Python 3.x
- LangChain
- Anthropic API
- dotenv (para gerenciar variáveis de ambiente)

## Como Clonar e Configurar o Projeto

### 1. Clonar o Repositório

```sh
# Clone o repositório do GitHub
git clone https://github.com/LarissaRLira/TextSummarizationApp
cd seu-repositorio
```

### 2. Criar e Ativar um Ambiente Virtual

No Windows (PowerShell):

```sh
python -m venv venv
venv\Scripts\activate
```

No Mac/Linux:

```sh
python -m venv venv
source venv/bin/activate
```

### 3. Instalar as Dependências

```sh
pip install -r requirements.txt
```

### 4. Criar um Arquivo `.env`

Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave da API da Anthropic:

```
ANTHROPIC_API_KEY=sua_chave_aqui
```

## Explicação do Código

### Importando Bibliotecas

```python
from langchain_anthropic import ChatAnthropic
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter  
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv, find_dotenv
import os
```

Essas bibliotecas são necessárias para carregar o modelo de IA, dividir o texto, criar documentos e carregar variáveis de ambiente.

### Carregando Variáveis de Ambiente

```python
load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
```

Aqui carregamos a chave da API armazenada no arquivo `.env`, garantindo que não seja exposta no código-fonte.

### Criando o Modelo de IA

```python
llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature= 0, # Ajusta o nível de criatividade do modelo
    anthropic_api_key = ANTHROPIC_API_KEY
)
```

Este trecho cria um modelo de IA da Anthropic chamado "Claude-3 Opus". O `temperature=0` garante que o modelo gere respostas mais determinísticas.

### Texto de Entrada

```python
text = "Caso precise colocar no PATH vai em Pesquisar -> variaveis de ambiente -> Editar variaveis de ambiente -> Ai vai abrir uma aba e vc clica em variaveis de ambiente"
```

Esse é o texto que será resumido.

### Dividindo o Texto

```python
text_splitter = CharacterTextSplitter()
texts = text_splitter.split_text(text)
```

Como alguns modelos lidam melhor com textos curtos, utilizamos o `CharacterTextSplitter` para dividir o texto em partes menores.

### Criando Documentos

```python
docs = [Document(page_content=text) for text in texts]
```

Aqui, transformamos cada parte do texto em um objeto `Document`, necessário para o LangChain processar corretamente a entrada.

### Criando a Cadeia de Resumo

```python
chain = load_summarize_chain(llm=llm, chain_type="stuff")
```

Criamos um pipeline (chain) para resumir os documentos usando o modelo de IA.

### Executando a Cadeia e Exibindo o Resultado

```python
summary = chain.invoke(docs)
print(summary['output_text'])
```

Chamamos a cadeia para gerar o resumo do texto e o imprimimos no console.

## Como Executar o Programa

Com o ambiente virtual ativado e os pacotes instalados, execute:

```sh
python app.py
```

Isso irá processar o texto e exibir um resumo gerado pelo modelo de IA.

## Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias no código!

## Licença

Este projeto está sob a licença MIT.

