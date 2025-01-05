import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("crustdata-index")

def generate_text(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in Crustdata API usage. Provide clear, concise, and accurate responses about using Crustdata's APIs for company and people data enrichment, search, and discovery. Dont always include code, first show first show the curl, for eg.how do i search for people given their current title,current comapny and location, answer:You can use api.crustdata.com/screener/person/search endpoint. Here is an example curl request to find “people with title engineer at OpenAI in San Francisco”  in this  or what is asked then How do I search for people given their current title, current company and location? show me just the curl file no python , explain API parameters, and suggest best practices. Always strive for clarity and precision in your answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )
    return response.choices[0].message.content

def query_pinecone(query, top_k=5):
    query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def get_response(user_input):
    pinecone_results = query_pinecone(user_input)
    context = "\n".join([result.metadata['text'] for result in pinecone_results.matches])

    prompt = f"""Context:
{context}

Query: {user_input}

Provide a clear and concise response to the query above, focusing on Crustdata API usage. Include the following elements in your answer:
1. A brief explanation of the relevant Crustdata API endpoint(s)
2. A code example (preferably in Python) demonstrating how to use the API, if applicable
3. Explanation of key parameters and their usage
4. Any potential limitations or considerations when using this API
5. Suggestions for best practices

Response:"""
    return generate_text(prompt)

if __name__ == "__main__":
    while True:
        user_input = input("Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = get_response(user_input)
        print(f"Response: {response}\n")
