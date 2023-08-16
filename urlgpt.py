import openai
import session_info
import gradio as gr
import sys
import os
import constants
from gpt_index import  LLMPredictor
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain


session_info.show()

os.environ["OPENAI_API_KEY"] = constants.APIKEY

def chatbot(input_text):
    num_outputs = 512  
   
    urls = [
        "https://www.plex.com/products/manufacturing-execution-system",
        "https://www.plex.com/smart-manufacturing-platform"
    ]
    
    loader = WebBaseLoader(urls)
    
    index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    result = chain({"question": input_text, "chat_history": chat_history}) 
    chat_history.append((input_text, result['answer']))   
    return result['answer']

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

iface.launch(share=True)