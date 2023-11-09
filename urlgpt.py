import openai
import gradio as gr
import sys
import os
import constants
import requests
from urllib.parse import urlsplit
import validators
import tldextract
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

os.environ["OPENAI_API_KEY"] = constants.APIKEY

def chatbot(query): 
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature="0.7"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history}) 
    chat_history.append((query, result['answer']))  

    return result['answer']

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
    
    loader = WebBaseLoader(childUrls)    
    index = VectorstoreIndexCreator().from_loaders([loader])

    iface.launch(share=True)