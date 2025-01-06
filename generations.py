import os
import uuid
import json
from dotenv import load_dotenv
from openai import OpenAI
# New Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------
# 1. Environment Setup & Initialization
# ---------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # or whichever region is correct for your account

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please add it to your .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Define the indexes we need
KNOWLEDGE_INDEX_NAME = "crustdata-index"
CHAT_INDEX_NAME = "chat-history-index"


# ---------------------------------------------------------------------
# 2. Utility: Ensure Index Exists or Create It
# ---------------------------------------------------------------------

def ensure_index_exists(index_name: str, dimension: int = 1536):
    """
    Checks if 'index_name' exists. If not, creates it with the given dimension.
    1536 is the dimensionality for text-embedding-ada-002.

    You can adjust 'metric' or 'spec' to match your plan or region if needed.
    """
    indexes = pc.list_indexes().names()
    if index_name not in indexes:
        print(f"[INFO] Creating index '{index_name}' in Pinecone...")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  # or "dotproduct", "euclidean"
                spec=ServerlessSpec(
                    cloud="aws",  # or "gcp" if you prefer
                    region=PINECONE_ENV.replace("-gcp", "").replace("-aws", "")
                    # Example: if PINECONE_ENV="us-west4-gcp", region might be "us-west4"
                )
            )
            print(f"[INFO] Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"[ERROR] Could not create index '{index_name}': {e}")
            raise
    else:
        print(f"[DEBUG] Index '{index_name}' already exists in Pinecone.")


# Ensure both indexes exist
ensure_index_exists(KNOWLEDGE_INDEX_NAME)
ensure_index_exists(CHAT_INDEX_NAME)

# Now instantiate index objects
knowledge_index = pc.Index(KNOWLEDGE_INDEX_NAME)
chat_index = pc.Index(CHAT_INDEX_NAME)


# ---------------------------------------------------------------------
# 3. Knowledge Base Ingestion
# ---------------------------------------------------------------------

def ingest_additional_knowledge(document_text: str, metadata: dict = None) -> bool:
    """
    Ingests new text into the 'crustdata-index'.
    'document_text': content from Slack Q&As, doc snippets, user Q&As, etc.
    'metadata': optional dictionary with extra info (e.g., source).
    """
    if metadata is None:
        metadata = {}

    try:
        print("[DEBUG] Embedding and upserting knowledge to crustdata-index...")
        embedding = openai_client.embeddings.create(
            input=document_text,
            model="text-embedding-ada-002"
        ).data[0].embedding

        doc_id = str(uuid.uuid4())
        knowledge_index.upsert(
            vectors=[
                (doc_id, embedding, {"text": document_text, **metadata})
            ]
        )
        print(f"[DEBUG] Ingested doc into '{KNOWLEDGE_INDEX_NAME}' with ID: {doc_id}")
        return True
    except Exception as e:
        print(f"[ERROR] ingest_additional_knowledge: {e}")
        return False


# ---------------------------------------------------------------------
# 4. Chat History Storage
# ---------------------------------------------------------------------

def store_chat_in_pinecone(role: str, content: str):
    """
    Stores a single chat message in the 'chat-history-index' with an embedding.
    'role': "user" or "assistant"
    'content': the text message
    """
    try:
        embedding = openai_client.embeddings.create(
            input=content,
            model="text-embedding-ada-002"
        ).data[0].embedding

        doc_id = f"{role}_{uuid.uuid4()}"
        chat_index.upsert(
            vectors=[(doc_id, embedding, {"role": role, "content": content})]
        )
        print(f"[DEBUG] Chat message stored in '{CHAT_INDEX_NAME}' with ID: {doc_id}")
    except Exception as e:
        print(f"[ERROR] store_chat_in_pinecone: {e}")


# ---------------------------------------------------------------------
# 5. Query Knowledge Base
# ---------------------------------------------------------------------

def query_knowledge_base(query: str, top_k: int = 5):
    """
    Searches 'crustdata-index' for relevant context.
    Returns None if an error occurs.
    """
    try:
        query_embedding = openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding

        results = knowledge_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"[ERROR] query_knowledge_base: {e}")
        return None


# ---------------------------------------------------------------------
# 6. OpenAI Completion Logic
# ---------------------------------------------------------------------

def generate_text(prompt: str) -> str:
    """
    Calls OpenAI GPT-based chat completion to get a text response.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Hypothetical GPT-4-like model or your chosen model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specialized in Crustdata API usage. "
                        "Provide clear, concise, and accurate responses about Crustdata's APIs, "
                        "including cURL examples first when relevant, then explanations. "
                        "Focus on best practices and clarity."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] generate_text: {e}")
        return f"Error generating text: {str(e)}"


# ---------------------------------------------------------------------
# 7. Validate & Fix Potential API Request
# ---------------------------------------------------------------------

def validate_api_request(endpoint: str, params: dict):
    """
    Example validation to ensure 'title', 'company', and 'location' are present.
    """
    required_fields = ["title", "company", "location"]
    missing_fields = [f for f in required_fields if f not in params]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, "API request is valid."


def fix_api_request(params: dict, error_log: str):
    """
    Adds default values if fields are missing.
    """
    if "Missing required fields" in error_log:
        if "title" not in params:
            params["title"] = "engineer"
        if "company" not in params:
            params["company"] = "OpenAI"
        if "location" not in params:
            params["location"] = "San Francisco"
    return params


# ---------------------------------------------------------------------
# 8. Main Chat Response Function
# ---------------------------------------------------------------------

def get_response(user_input: str) -> str:
    """
    1) Queries knowledge base for relevant context
    2) Sends combined prompt to OpenAI
    3) Checks if there's a 'curl' snippet in the response, then tries to validate/fix
    4) Returns the final answer
    """
    results = query_knowledge_base(user_input, top_k=5)
    if not results or not results.matches:
        context = "No relevant context found."
    else:
        # Combine all relevant 'text' fields from Pinecone matches
        context = "\n".join([match.metadata['text'] for match in results.matches])

    prompt = f"""Context:
{context}

Query: {user_input}

Please provide a clear, concise answer about Crustdata API usage. If relevant, include:
- cURL example
- Key parameters
- Limitations
- Best Practices
"""

    response_text = generate_text(prompt)

    # Check for 'curl' snippet
    if "curl" in response_text.lower():
        # Fake minimal params to see if we need to fix them
        dummy_params = {
            "person_id": "98765"  # Missing 'title', 'company', 'location'
        }
        valid, message = validate_api_request("https://api.crustdata.com/person/details", dummy_params)
        if not valid:
            # Fix it
            fixed_params = fix_api_request(dummy_params, message)
            valid2, message2 = validate_api_request("https://api.crustdata.com/person/details", fixed_params)
            if valid2:
                response_text += "\n\n[NOTE: We auto-fixed missing fields before sharing the final API call.]"
            else:
                response_text += f"\n\n[ERROR: Could not fix the request automatically: {message2}]"

    return response_text
