from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk

nltk.download("punkt")

# === Load and split resume ===
loader = TextLoader("./data/resume.txt")
docs = loader.load()

# Optional: Split into smaller chunks if needed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# === Embeddings & Vectorstore ===
embedding_model_name = "multi-qa-MiniLM-L6-cos-v1"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")

# === LLM ===
hf_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")
llm = HuggingFacePipeline(pipeline=hf_pipe)

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a virtual version of a person named Shivam. You may be asked questions in the first or third person.
ONLY use the context below to answer. If unsure, say "I don’t know."

Context:
{context}

Question: {question}
Answer:"""
)

# === QA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt_template}
)

# === FastAPI ===
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    try:
        response = qa_chain.run(query.question)
        return {"answer": response.strip()}
    except Exception as e:
        return {"error": str(e)}
