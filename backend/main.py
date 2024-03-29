from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.params import Form
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os

UPLOAD_DIR = 'C:/Users/User/react/backend/uploads'

app = FastAPI()

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"]    

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
# embeddings = OpenAIEmbeddings()

import json

def message_to_dict(message):
    return {'content': message.content, 'type': type(message).__name__}

def get_pdf_text(path):
    text = ""
    
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.post('/uploadfile/')
async def create_upload_file(file_upload:UploadFile):
    data = await file_upload.read()
    save_to = Path(UPLOAD_DIR) / file_upload.filename
    with open(save_to,'wb') as f:
        f.write(data)
    return {"filenames": file_upload.filename}


@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    try:
        # Get PDF text
        pdf_files = [str(file) for file in Path(UPLOAD_DIR).iterdir() if file.is_file() and file.suffix == '.pdf']
        print(pdf_files[0])
        raw_text = get_pdf_text(pdf_files[0])
        # Get text chunks
        text_chunks = get_text_chunks(raw_text)
        # Create vector store
        vectorstore = get_vectorstore(text_chunks)
        # Create conversation chain
        conversation_chain = get_conversation_chain(vectorstore)
        # Get response
        response = conversation_chain({'question': question})
        chat_history = response['chat_history']
        chat_history_dicts = [message_to_dict(message) for message in chat_history]
        chat_history_json = json.dumps(chat_history_dicts)
        print(chat_history_json)
        return JSONResponse(content={"chat_history": chat_history_json})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)








# import logging
# from fastapi import FastAPI, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# from fastapi.params import Form
# import uvicorn
# from dotenv import load_dotenv
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# import os

# # Configure logging
# logging.basicConfig(level=logging.ERROR)

# UPLOAD_DIR = 'C:/Users/User/react/backend/uploads'

# app = FastAPI()

# load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"]    

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*']
# )

# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
# # embeddings = OpenAIEmbeddings()

# import json

# def message_to_dict(message):
#     return {'content': message.content, 'type': type(message).__name__}

# def get_pdf_text(path):
#     text = ""
    
#     pdf_reader = PdfReader(path)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     # llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# @app.post('/uploadfile/')
# async def create_upload_file(file_upload:UploadFile):
#     try:
#         data = await file_upload.read()
#         save_to = Path(UPLOAD_DIR) / file_upload.filename
#         with open(save_to,'wb') as f:
#             f.write(data)
#         return {"filenames": file_upload.filename}
#     except Exception as e:
#         logging.error(f"Error uploading file: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# @app.post("/query")
# async def query_endpoint(question: str = Form(...)):
#     try:
#         # Get PDF text
#         pdf_files = [str(file) for file in Path(UPLOAD_DIR).iterdir() if file.is_file() and file.suffix == '.pdf']
#         print(pdf_files[0])
#         raw_text = get_pdf_text(pdf_files[0])
#         # Get text chunks
#         text_chunks = get_text_chunks(raw_text)
#         # Create vector store
#         vectorstore = get_vectorstore(text_chunks)
#         # Create conversation chain
#         conversation_chain = get_conversation_chain(vectorstore)
#         # Get response
#         response = conversation_chain({'question': question})
#         chat_history = response['chat_history']
#         chat_history_dicts = [message_to_dict(message) for message in chat_history]
#         chat_history_json = json.dumps(chat_history_dicts)
#         print(chat_history_json)
#         return JSONResponse(content={"chat_history": chat_history_json})
#     except Exception as e:
#         logging.error(f"Error processing query: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
