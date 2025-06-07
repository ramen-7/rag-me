from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Load and chunk your resume/bio
with open("data/resume.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2") 

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="personal-info",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)


for i, chunk in enumerate(documents):
    collection.add(documents=[chunk], ids=[f"chunk_{i}"])

# Load LLM
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

# FastAPI setup
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    results = collection.query(query_texts=[query.question], n_results=3)
    context = "\n".join(results["documents"][0])

    prompt = f"""You are a helpful assistant. ONLY use the context below to answer the question. 
        If the answer is not in the context, say "I don’t know."

        Context:
        {context}

        Question: {query.question}
        Answer:"""


    output = llm(prompt, max_new_tokens=150)[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()
    return {"answer": answer}
