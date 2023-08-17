import openai
import gradio as gr
import sys
import os
import constants
import requests
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"] = constants.APIKEY

def chatbot(input_text):
    #url = 'https://www.plex.com/smart-manufacturing-platform'

    #urls = getchildurl(url)

    urls = [
        'https://www.plex.com/smart-manufacturing-platform',
        'https://www.plex.com/products/manufacturing-execution-system',
        'https://www.plex.com/industries/food-and-beverage/why-mes-critical-food-and-beverage-manufacturers'
    ]
    
    loader = WebBaseLoader(urls)
    
    index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature="0.7"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    result = chain({"question": input_text, "chat_history": chat_history}) 
    chat_history.append((input_text, result['answer']))   
    return result['answer']


def getchildurl (url):
    base_add = 'https://www.plex.com'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = []
    for link in soup.find_all('a'):
        url = link.get('href')
        if (url != None and url != '' and url != '#'):
            if "https:" in url:
                urls.append(url)
            else:
                url = base_add + url
                urls.append(url)
    
    return urls


iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query"),
                     outputs="text",
                     title="DocuConverse: Empowering Conversations with Knowledge Docs",
                     allow_flagging='never',
                     flagging_options=None)

iface.launch(share=True)