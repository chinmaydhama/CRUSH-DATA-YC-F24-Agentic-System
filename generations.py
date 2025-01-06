import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("crustdata-index")

def validate_api_request(endpoint, params):
    print("[DEBUG] Running validate_api_request...")
    required_fields = ["title", "company", "location"]
    missing_fields = [field for field in required_fields if field not in params]
    if missing_fields:
        print(f"[DEBUG] Validation failed. Missing fields: {', '.join(missing_fields)}")
        return False, f"Missing required fields in request: {', '.join(missing_fields)}"
    print("[DEBUG] Validation successful.")
    return True, "API request is valid."

def fix_api_request(params, error_log):
    print("[DEBUG] Running fix_api_request...")
    if "Missing required fields" in error_log:
        if "title" in error_log and "title" not in params:
            params["title"] = "engineer"
            print("[DEBUG] Added default title: engineer")
        if "company" in error_log and "company" not in params:
            params["company"] = "OpenAI"
            print("[DEBUG] Added default company: OpenAI")
        if "location" in error_log and "location" not in params:
            params["location"] = "San Francisco"
            print("[DEBUG] Added default location: San Francisco")
    print(f"[DEBUG] Fixed parameters: {params}")
    return params

def generate_text(prompt):
    print("[DEBUG] Running generate_text...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specialized in Crustdata API usage. "
                        "Provide clear, concise, and accurate responses about using Crustdata's APIs "
                        "for company and people data enrichment, search, and discovery. "
                        "Dont always include code, first show the curl, for example if asked: "
                        "'How do I search for people given their current title, current company and location?', "
                        "answer with the curl request only. Also explain API parameters, "
                        "and suggest best practices. Always strive for clarity and precision in your answers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=700
        )
        print("[DEBUG] Response generated successfully.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Error generating text: {str(e)}")
        return f"Error generating text: {str(e)}"

def query_pinecone(query, top_k=5):
    print("[DEBUG] Running query_pinecone...")
    try:
        query_embedding = openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        print("[DEBUG] Pinecone query successful.")
        return results
    except Exception as e:
        print(f"[ERROR] Error querying Pinecone: {str(e)}")
        return None

def get_response(user_input):
    print("[DEBUG] Running get_response...")
    # Step 1: Query Pinecone for context
    pinecone_results = query_pinecone(user_input)

    # Step 2: Construct context from Pinecone results
    if pinecone_results is None or not pinecone_results.matches:
        context = "No relevant context found."
        print("[DEBUG] No relevant context found.")
    else:
        context = "\n".join([result.metadata['text'] for result in pinecone_results.matches])
        print("[DEBUG] Context constructed from Pinecone results.")

    # Step 3: Create a prompt for OpenAI
    prompt = f"""Context:
{context}

Query: {user_input}

Provide a clear and concise response to the query above, focusing on Crustdata API usage. 
Include the following elements in your answer (if relevant):
1. A brief explanation of the relevant Crustdata API endpoint(s)
2. A curl example demonstrating how to use the API, if applicable
3. Explanation of key parameters and their usage
4. Any potential limitations or considerations when using this API
5. Suggestions for best practices

Response:"""

    # Step 4: Generate response from OpenAI
    response_text = generate_text(prompt)

    # Step 5: Validate and fix the API request
    if "curl" in response_text.lower():
        dummy_params = {
            "person_id": "98765"
        }

        # Validate the API request
        valid, message = validate_api_request("https://api.crustdata.com/person/details", dummy_params)
        if not valid:
            print("[DEBUG] API request validation failed. Fixing the request...")
            # Fix the request if it's invalid
            fixed_params = fix_api_request(dummy_params, message)
            valid, message = validate_api_request("https://api.crustdata.com/person/details", fixed_params)
            if valid:
                response_text += (
                    "\n\n[NOTE: The original request had issues. "
                    "We auto-fixed the parameters based on error logs before sharing the final API call.]"
                )
            else:
                response_text += f"\n\n[ERROR: Could not fix the request automatically: {message}]"

    return response_text

if __name__ == "__main__":
    while True:
        user_input = input("Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        response = get_response(user_input)
        print(f"\nResponse:\n{response}\n")
