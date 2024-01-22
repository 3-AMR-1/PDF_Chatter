import torch
import streamlit as st
import PyPDF2 as pypdf
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text= ""
    for pdf in pdf_docs:
        pdf_reader = pypdf.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 1000,
        chunk_overlap= 300,     #retain meaning with sentence overlap
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    #allows html to be parsed in the website
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    

def main():
    #extract API keys from .env -- for langchain
    load_dotenv()
    
    #creates a simple local host website
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    #initialises session state for chat history
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
     
    st.header("Chat with PDFs :books:")
    user_question = st.text_input("What would you like to know: ")
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner ("Processing"): #creates a spinning wheel during load
                #retrieve pdf
                raw_text = get_pdf_text(pdf_docs)
                
                #retrieve text chunks
                text_chunks = get_text_chunks(raw_text)

                #establish vector store
                vectorstore = get_vectorstore(text_chunks)
                
                #conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)     #initialises outside the sidebar
                
      

#check on application execution rather than importing 
if __name__ == '__main__':
    main()
    
#streamlit run c:\Users\ammar\OneDrive\VS_CODE\PDF_Chatter\app.py