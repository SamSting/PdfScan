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
# from langchain_community.vectorstores import FAISS

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# import json

# import os
# import warnings
# import logging

# # Suppress all DeprecationWarnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World!"}


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
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# @app.post('/uploadfile/')
# async def create_upload_file(file_upload:UploadFile):
#     data = await file_upload.read()
#     save_to = Path(UPLOAD_DIR) / file_upload.filename
#     with open(save_to,'wb') as f:
#         f.write(data)
#     return {"filenames": file_upload.filename}


# logger = logging.getLogger(__name__)

# @app.post("/query")
# async def query_endpoint(question: str = Form(...)):
#     try:
#         pdf_files = [str(file) for file in Path(UPLOAD_DIR).iterdir() if file.is_file() and file.suffix == '.pdf']
        
#         if not pdf_files:
#             raise FileNotFoundError("No PDF files found in the upload directory.")
        
#         raw_text = get_pdf_text(pdf_files[0])
#         text_chunks = get_text_chunks(raw_text)
#         vectorstore = get_vectorstore(text_chunks)
        
#         with suppress_langchain_deprecation_warning():
#             conversation_chain = get_conversation_chain(vectorstore)
#             response = conversation_chain.invoke({'question': question})
        
#         chat_history = response['chat_history']
#         chat_history_dicts = [message_to_dict(message) for message in chat_history]
#         chat_history_json = json.dumps(chat_history_dicts)
        
#         logger.info("Query successful.")
#         return JSONResponse(content={"chat_history": chat_history_json})
    
#     except FileNotFoundError as e:
#         logger.error(f"File not found error: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=404)
    
#     except Exception as e:
#         logger.exception("An unexpected error occurred.")
#         return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter  # Ensure this import is correct
import json
import os
import warnings
import logging

# Suppress all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Define constants
UPLOAD_DIR = 'C:/Users/User/react/backend/uploads'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

# Logger setup
logger = logging.getLogger(__name__)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

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
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    save_to = Path(UPLOAD_DIR) / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    return {"filenames": file_upload.filename}

@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    try:
        pdf_files = [str(file) for file in Path(UPLOAD_DIR).iterdir() if file.is_file() and file.suffix == '.pdf']
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in the upload directory.")
        
        raw_text = get_pdf_text(pdf_files[0])
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        
        conversation_chain = get_conversation_chain(vectorstore)
        response = conversation_chain.invoke({'question': question})
        
        chat_history = response['chat_history']
        chat_history_dicts = [message_to_dict(message) for message in chat_history]
        chat_history_json = json.dumps(chat_history_dicts)
        
        logger.info("Query successful.")
        return JSONResponse(content={"chat_history": chat_history_json})
    
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=404)
    
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
