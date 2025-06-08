from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from chromadb import Client
from chromadb.utils import embedding_functions
from nltk.tokenize import sent_tokenize
import nltk

# Download sentence tokenizer
nltk.download('punkt')

# === Load and chunk resume text ===
with open("./data/resume.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Split into semantic chunks based on double newlines
documents = full_text.split("\n\n")

# === Setup embedding and ChromaDB ===
embed_model_id = "multi-qa-MiniLM-L6-cos-v1"
embedder = SentenceTransformer(embed_model_id)

chroma_client = Client()
collection = chroma_client.create_collection(
    name="personal-info",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(embed_model_id)
)

# Add document chunks to ChromaDB
for i, chunk in enumerate(documents):
    collection.add(documents=[chunk], ids=[f"chunk_{i}"])

# === Load LLM ===
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

# === FastAPI setup ===
app = FastAPI()

class Query(BaseModel):
    question: str

# Filter chunks by context type based on question
def filter_by_scope_json(resume_json, question: str):
    q = question.lower()
    results = {}

    if "project" in q and "work" in q:
        # include both experience and projects
        results["EXPERIENCE"] = resume_json.get("EXPERIENCE", [])
        results["PROJECTS"] = resume_json.get("PROJECTS", [])
    elif any(kw in q for kw in ["project", "projects", "built", "developed"]):
        results["PROJECTS"] = resume_json.get("PROJECTS", [])
    elif any(kw in q for kw in ["work", "company", "job", "experience", "employer"]):
        results["EXPERIENCE"] = resume_json.get("EXPERIENCE", [])
    elif any(kw in q for kw in ["certification", "certified"]):
        results["CERTIFICATIONS"] = resume_json.get("CERTIFICATIONS", [])
    elif any(kw in q for kw in ["profile", "leetcode", "github", "kaggle"]):
        results["PROFILES"] = resume_json.get("PROFILES", [])
    elif any(kw in q for kw in ["interest", "hobby", "passion"]):
        results["INTERESTS"] = resume_json.get("INTERESTS", [])
    elif any(kw in q for kw in ["contact", "email", "phone", "linkedin"]):
        results["CONTACT"] = resume_json.get("CONTACT", [])
    else:
        # default fallback — return everything
        results = resume_json

    return results



@app.post("/ask")
def ask_question(query: Query):
    # Narrow the documents using scope detection
    scoped_docs = filter_by_scope(documents, query.question)

    # Temporary collection for scoped documents
    temp_collection = chroma_client.get_or_create_collection(
        name="temp-scope",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(embed_model_id)
    )
    for i, doc in enumerate(scoped_docs):
        temp_collection.add(documents=[doc], ids=[f"temp_chunk_{i}"])

    results = temp_collection.query(query_texts=[query.question], n_results=3)
    context = "\n".join(doc for docs in results["documents"] for doc in docs)

    print(f"\n[Query]: {query.question}")
    print(f"[Retrieved Context]:\n{context}\n")

    prompt = f"""You are a virtual version of a person named Shivam. You may be asked questions in either the first person ("What are your skills?") or the third person ("What are Shivam's skills?"). 
    ONLY use the context provided below to answer the question. If the answer is not in the context, respond with "I don’t know."

    Context:
    {context.strip()}

    Question: {query.question.strip()}
    Answer:"""

    output = llm(prompt, max_new_tokens=1024)[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()

    return {"answer": answer}
