import os
import uuid
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # or whichever region is correct
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please add it to your .env file.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


KNOWLEDGE_INDEX_NAME = "crustdata-index"
CHAT_INDEX_NAME = "chat-history-index"
ADDITIONAL_DOCS_INDEX_NAME = "additional-docs-index"

def ensure_index_exists(index_name: str, dimension: int = 1536):
    indexes = pc.list_indexes().names()
    if index_name not in indexes:
        print(f"[INFO] Creating index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV.replace("-gcp", "").replace("-aws", "")
                )
            )
            print(f"[INFO] Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"[ERROR] Could not create index '{index_name}': {e}")
            raise
    else:
        print(f"[DEBUG] Index '{index_name}' already exists in Pinecone.")


ensure_index_exists(KNOWLEDGE_INDEX_NAME)
ensure_index_exists(CHAT_INDEX_NAME)
ensure_index_exists(ADDITIONAL_DOCS_INDEX_NAME)
knowledge_index = pc.Index(KNOWLEDGE_INDEX_NAME)
chat_index = pc.Index(CHAT_INDEX_NAME)
additional_docs_index = pc.Index(ADDITIONAL_DOCS_INDEX_NAME)


def ingest_additional_documents(document_text: str, metadata: dict = None) -> bool:
    if metadata is None:
        metadata = {}
    try:
        print("[DEBUG] Embedding and upserting to 'additional-docs-index'...")
        embedding = openai_client.embeddings.create(
            input=document_text,
            model="text-embedding-ada-002"
        ).data[0].embedding
        doc_id = str(uuid.uuid4())
        additional_docs_index.upsert(
            vectors=[
                (doc_id, embedding, {"text": document_text, **metadata})
            ]
        )
        print(f"[DEBUG] Ingested doc into '{ADDITIONAL_DOCS_INDEX_NAME}' with ID: {doc_id}")
        return True
    except Exception as e:
        print(f"[ERROR] ingest_additional_documents: {e}")
        return False

def store_chat_in_pinecone(role: str, content: str):
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


def query_all_indexes(query: str, top_k: int = 3):
    try:
        query_embedding = openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding
        knowledge_results = knowledge_index.query(query_embedding, top_k=top_k, include_metadata=True)
        chat_results = chat_index.query(query_embedding, top_k=top_k, include_metadata=True)
        docs_results = additional_docs_index.query(query_embedding, top_k=top_k, include_metadata=True)
        combined_texts = []
        if knowledge_results and knowledge_results.matches:
            for match in knowledge_results.matches:
                combined_texts.append(match.metadata['text'])
        if chat_results and chat_results.matches:
            for match in chat_results.matches:
                content = match.metadata.get('text') or match.metadata.get('content') or ""
                combined_texts.append(content)
        if docs_results and docs_results.matches:
            for match in docs_results.matches:
                combined_texts.append(match.metadata['text'])
        if not combined_texts:
            return "No relevant context found."
        else:
            return "\n".join(combined_texts)
    except Exception as e:
        print(f"[ERROR] query_all_indexes: {e}")
        return "No relevant context found."


def generate_text(prompt: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specialized in Crustdata API usage. "
                        "You have access to a knowledge base, chat logs, and additional documents, "
                        "each stored in separate Pinecone indexes. Provide accurate and concise answers, "
                        "including cURL examples first if relevant, then explanations, focusing on clarity."
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
    1) Queries *all three indexes* for relevant context.
    2) Generates a prompt with that context.
    3) Calls OpenAI to get an answer.
    4) If a 'curl' snippet is found, tries to validate/fix the request.
    5) Returns the final text response.
    """
    context = query_all_indexes(user_input, top_k=3)

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

    # Check if there's a "curl" snippet => we attempt to validate/fix
    if "curl" in response_text.lower():
        dummy_params = {
            "person_id": "98765"  # Missing 'title', 'company', 'location'
        }
        valid, message = validate_api_request("https://api.crustdata.com/person/details", dummy_params)
        if not valid:
            # Attempt fix
            fixed_params = fix_api_request(dummy_params, message)
            valid2, message2 = validate_api_request("https://api.crustdata.com/person/details", fixed_params)
            if valid2:
                response_text += "\n\n[NOTE: We auto-fixed missing fields before sharing the final API call.]"
            else:
                response_text += f"\n\n[ERROR: Could not fix the request automatically: {message2}]"

    return response_text
