import openai
import gradio as gr
import sys
import os
import constants
import requests
from urllib.parse import urlsplit
import validators
import tldextract
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

os.environ["OPENAI_API_KEY"] = constants.APIKEY

def chatbot(query): 
    retriever = vectorstore.as_retriever()
    rag_prompt = hub.pull("rlm/rag-prompt")   

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature="0.7")
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm

    result = rag_chain.invoke(query)
    
    return result.content

# Get all child url present on page
def getchildurl (url):
    base_add = 'https://www.plex.com'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = []
    for link in soup.find_all('a'):
        url = link.get('href')
        if (url != None and url != '' and url != '#'):            
            if "https:" in url:
                if is_valid_url(url):
                    urls.append(url)
            else:
                url = base_add + url
                if is_valid_url(url):
                    urls.append(url)   
    return urls

# Check valid URL
def is_valid_url(url):
    extracted = tldextract.extract(url)
    return validators.url(url) and bool(extracted.domain) and bool(extracted.suffix)


# Load URL using Web based loader
def webbasedloader(childUrls): 
    loader = WebBaseLoader(childUrls)    
    data = loader.load()

    return data

# Load URL using Asynchronous html loader
def htmlloader(childUrls):
    loader = AsyncHtmlLoader(childUrls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    return docs_transformed


# Gradio interface for user input
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query"),
                     outputs="text",
                     title="DocuConverse: Empowering Conversations with Knowledge Docs",
                     allow_flagging='never',
                     flagging_options=None)


if __name__ == '__main__':
    # url = 'https://www.plex.com/smart-manufacturing-platform'
    urls = [
        'https://www.plex.com/smart-manufacturing-platform',
        'https://www.plex.com/products/manufacturing-execution-system',
        'https://www.plex.com/industries/food-and-beverage/why-mes-critical-food-and-beverage-manufacturers'
    ]

    uniqueUrl = set()
    for url in urls:
        uniqueUrl.update(getchildurl(url))

    childUrls = list(uniqueUrl)

    data = webbasedloader(childUrls)
    #data = htmlloader(childUrls)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)    
    splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    iface.launch(share=True)