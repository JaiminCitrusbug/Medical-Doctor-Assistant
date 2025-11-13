import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-vectors")

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for query."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def retrieve_similar_chunks(query, top_k=3):
    """Retrieve top-k similar chunks for the given query using Pinecone."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    query_embedding = get_embedding(query)
    
    # Initialize Pinecone and connect to index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(name=PINECONE_INDEX_NAME)
    
    # Query Pinecone for similar vectors
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results to match the original structure
    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata.get("text", ""),
            "similarity": float(match.score)  # Pinecone returns cosine similarity as score
        })
    
    return chunks

