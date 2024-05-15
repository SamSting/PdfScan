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
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
import json

import os
import warnings
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level (optional)
logger.setLevel(logging.INFO)

# Define a handler to output logs to the console (you can also configure file-based logging)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Define a formatter to format the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


# Suppress all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Define the directory to store uploads
UPLOAD_DIR = 'C:/Users/User/react/backend/uploads'

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

# Initialize the Hugging Face LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

# Define function to convert message to dictionary
def message_to_dict(message):
    return {'content': message.content, 'type': type(message).__name__}

# Define function to extract text from PDF
def get_pdf_text(path):
    text = ""
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Define function to generate vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Define function to create conversational chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Define endpoint to handle file upload
@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    save_to = Path(UPLOAD_DIR) / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    return {"filenames": file_upload.filename}

# Define endpoint to handle queries
@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    try:
        # Get PDF files in the upload directory
        pdf_files = [str(file) for file in Path(UPLOAD_DIR).iterdir() if file.is_file() and file.suffix == '.pdf']
        
        # Raise error if no PDF files found
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in the upload directory.")
        
        # Extract text from the first PDF file
        raw_text = get_pdf_text(pdf_files[0])
        
        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Generate vectorstore from text chunks
        vectorstore = get_vectorstore(text_chunks)
        
        # Create conversational chain
        conversation_chain = get_conversation_chain(vectorstore)
        
        # Invoke conversation chain with the question
        response = conversation_chain.invoke({'question': question})
        
        # Convert chat history to JSON format
        chat_history = response['chat_history']
        chat_history_dicts = [message_to_dict(message) for message in chat_history]
        chat_history_json = json.dumps(chat_history_dicts)
        
        # Log query success
        logger.info("Query successful.")
        
        # Return chat history as JSON response
        return JSONResponse(content={"chat_history": chat_history_json})
    
    except FileNotFoundError as e:
        # Log file not found error
        logger.error(f"File not found error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=404)
    
    except Exception as e:
        # Log unexpected error
        logger.exception("An unexpected error occurred.")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
