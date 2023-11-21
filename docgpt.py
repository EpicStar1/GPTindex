from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 1
    chunk_size_limit = 600  
   
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)  
    #Directory in which the indexes will be stored  
    index.storage_context.persist(persist_dir="indexes")

    #print(index)
    return index

def chatbot(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")    
    #load indexes from directory using storage_context 
    query_engne = load_index_from_storage(storage_context).as_query_engine()    
    response = query_engne.query(input_text)    
    #returning the response
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query"),
                     outputs="text",
                     title="DocuConverse: Empowering Conversations with Knowledge Docs",
                     allow_flagging='never',
                     flagging_options=None)

index = construct_index("docs")
iface.launch(share=True)