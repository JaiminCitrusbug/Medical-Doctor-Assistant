# vector_store.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-vectors")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

INPUT_JSON = os.getenv("INPUT_JSON", "wockhardt_products.json")

def build_text_for_embedding(doc: dict) -> str:
    """
    Construct a single text blob from the Wockhardt product record.
    Uses available fields safely (some keys may not exist).
    """
    fields_in_order = [
        "product_name", "brand_name", "therapeutic_class", "strength",
        "dosage_form", "pack_size", "composition", "indication_summary",
        "extracted_text"
    ]
    parts = []
    for k in fields_in_order:
        v = doc.get(k)
        if v:
            parts.append(f"{k.replace('_',' ').title()}: {v}")
    # Fallback if nothing present
    if not parts:
        parts.append(doc.get("id", ""))
    text_blob = "\n".join(parts).strip()
    # keep it within a reasonable size for embeddings
    return text_blob[:6000]

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def get_aws_region(region_string: str):
    """Get AwsRegion enum value from string, with fallback."""
    # Map of common region strings to enum values
    region_map = {
        "us-east-1": AwsRegion.US_EAST_1,
        "us-west-2": AwsRegion.US_WEST_2,
        "eu-west-1": AwsRegion.EU_WEST_1,
    }
    
    # Try to get from map first
    if region_string in region_map:
        return region_map[region_string]
    
    # Try to find enum value dynamically (e.g., "ap-southeast-1" -> "AP_SOUTHEAST_1")
    try:
        # Convert "ap-southeast-1" to "AP_SOUTHEAST_1" format
        enum_name = region_string.replace("-", "_").upper()
        if hasattr(AwsRegion, enum_name):
            return getattr(AwsRegion, enum_name)
    except Exception:
        pass
    
    # Default to US_EAST_1 if not found
    print(f"⚠️  Region '{region_string}' not found, defaulting to us-east-1")
    return AwsRegion.US_EAST_1

def store_embeddings():
    """Store embeddings in Pinecone index."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Get AWS region enum value
        aws_region = get_aws_region(PINECONE_REGION)
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=aws_region
            )
        )
        print(f"✅ Index {PINECONE_INDEX_NAME} created successfully with dimension {embedding_dimension}!")
    else:
        print(f"✅ Index {PINECONE_INDEX_NAME} already exists")
        # Check index dimension
        index_stats = pc.describe_index(PINECONE_INDEX_NAME)
        index_dimension = index_stats.dimension
        
        if index_dimension != embedding_dimension:
            raise ValueError(
                f"❌ Dimension mismatch!\n"
                f"   Index '{PINECONE_INDEX_NAME}' has dimension: {index_dimension}\n"
                f"   Embedding model '{EMBEDDING_MODEL}' produces dimension: {embedding_dimension}\n\n"
                f"   Solutions:\n"
                f"   1. Use a different index name (set PINECONE_INDEX_NAME in .env)\n"
                f"   2. Delete the existing index and recreate it\n"
                f"   3. Use an embedding model that matches the index dimension\n"
            )
        else:
            print(f"✅ Index dimension ({index_dimension}) matches embedding dimension ({embedding_dimension})")
    
    # Connect to index
    index = pc.Index(name=PINECONE_INDEX_NAME)
    
    # Load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # Prepare vectors for batch upsert
    vectors_to_upsert = []
    
    for i, doc in enumerate(docs):
        uid = doc.get("id", f"vec_{i}")
        title = doc.get("product_name") or doc.get("brand_name") or uid
        source_id = doc.get("source_url", "")
        text = build_text_for_embedding(doc)
        embedding = create_embedding(text)
        
        # Pinecone metadata can store additional info
        metadata = {
            "title": title,
            "source_id": source_id,
            "chunk_index": 0,
            "text": text
        }
        
        vectors_to_upsert.append({
            "id": uid,
            "values": embedding,
            "metadata": metadata
        })
    
    # Batch upsert to Pinecone (upsert in batches of 100)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
    
    print("✅ All embeddings stored successfully in Pinecone!")

if __name__ == "__main__":
    store_embeddings()

