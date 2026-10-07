import os
import sys
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
os.environ.setdefault("USER_AGENT", "RAG-App/1.0")

#LangChain RAG Pipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# LLM setup
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(model = "openai/gpt-oss-20b", api_key=api_key, temperature=0)

# loading Resume
def load_docs(source: str):
    if source.endswith(".pdf"):
        return PyPDFLoader(source).load()
    return TextLoader(source).load()


# Chunking Resume
def split_docs(docs, chunk_size = 1000, chunk_overlap = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)


#Loading Chunks to Vector-DB
def build_store(chunks, persist_dir = "chroma_db"):
    emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") # confirm the naming of model
    return Chroma.from_documents(chunks, emb, persist_directory=persist_dir)


#Formating retrieved list of chunks
def format_docs(docs):
    return "|\n\n".join(d.page_content for d in docs)


#Prompt for LLM
PROMPT = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer based only on the above. Never answer beyond it, if you don't know, say "I don't know" """)

#Chain using LCEL
def build_chain(store, k=4):
    retriever = store.as_retriever(search_kwargs = {"k":k})
    return(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )

if __name__ == "__main__":
    store = build_store(split_docs(load_docs("Kesavadas_AI_Engineer_Resume.pdf")))
    chain = build_chain(store)

    question = "What are the tech stacks in this resume"
    print(f"Question:{question}\n")
    print("Answer:")
    print(chain.invoke(question))
