"""
Helper script to manage Pinecone indexes.
Use this to list, describe, or delete indexes.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)

def list_indexes():
    """List all indexes."""
    print("📋 Available Pinecone Indexes:\n")
    indexes = pc.list_indexes()
    
    if not indexes:
        print("   No indexes found.")
        return
    
    for idx in indexes:
        try:
            stats = pc.describe_index(idx.name)
            print(f"   Name: {idx.name}")
            print(f"   Dimension: {stats.dimension}")
            print(f"   Metric: {stats.metric}")
            print(f"   Status: {stats.status.get('ready', 'unknown') if hasattr(stats.status, 'get') else stats.status}")
            print()
        except Exception as e:
            print(f"   Name: {idx.name}")
            print(f"   Error getting details: {e}")
            print()

def describe_index(index_name: str):
    """Describe a specific index."""
    try:
        stats = pc.describe_index(index_name)
        print(f"📊 Index Details for '{index_name}':\n")
        print(f"   Dimension: {stats.dimension}")
        print(f"   Metric: {stats.metric}")
        print(f"   Status: {stats.status.get('ready', 'unknown') if hasattr(stats.status, 'get') else stats.status}")
        
        # Try to get index stats
        try:
            index = pc.Index(name=index_name)
            index_stats = index.describe_index_stats()
            print(f"   Total Vectors: {index_stats.get('total_vector_count', 'N/A')}")
        except Exception as e:
            print(f"   Could not get vector count: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def delete_index(index_name: str, confirm: bool = False):
    """Delete an index."""
    if not confirm:
        print(f"⚠️  To delete index '{index_name}', run with confirm=True")
        print(f"   Example: delete_index('{index_name}', confirm=True)")
        return
    
    try:
        pc.delete_index(index_name)
        print(f"✅ Index '{index_name}' deleted successfully!")
    except Exception as e:
        print(f"❌ Error deleting index: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_indexes()
        elif command == "describe" and len(sys.argv) > 2:
            describe_index(sys.argv[2])
        elif command == "delete" and len(sys.argv) > 2:
            index_name = sys.argv[2]
            confirm = len(sys.argv) > 3 and sys.argv[3].lower() == "confirm"
            delete_index(index_name, confirm)
        else:
            print("Usage:")
            print("  python manage_indexes.py list")
            print("  python manage_indexes.py describe <index_name>")
            print("  python manage_indexes.py delete <index_name> confirm")
    else:
        list_indexes()
        print("\nUsage:")
        print("  python manage_indexes.py list")
        print("  python manage_indexes.py describe <index_name>")
        print("  python manage_indexes.py delete <index_name> confirm")

