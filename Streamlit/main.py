import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

apiKey = os.environ.get("OPENAI_API_KEY")
if not apiKey:
    raise ValueError("API_KEY environment variable is not set.")


# Load PDF
def loadPdf(pdfName):
    folderPath = os.environ.get("PDF_PATH")

    if not folderPath:
        raise ValueError("PDF_PATH environment variable is not set.")

    pdfPath = os.path.join(folderPath, pdfName)

    loader = PyPDFLoader(pdfPath)

    docs = loader.load()

    return docs

# Create embeddings and vector store 
def createVectorStore(docs):
    textSplitter = RecursiveCharacterTextSplitter()
    chunks = textSplitter.split_documents(docs)

    embedding = OpenAIEmbeddings(model= "text-embedding-3-large")
    vectorStore = FAISS.from_documents(chunks, embedding)

    return vectorStore

# Create retrieval chain 
def createChain(vectorStore):
    model = ChatOpenAI()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    documentChain = create_stuff_documents_chain(model, prompt)
