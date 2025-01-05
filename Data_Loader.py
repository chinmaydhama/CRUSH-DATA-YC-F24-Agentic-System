import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm
import pymupdf

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "crustdata-index"

# Check if index exists, if not create it
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )

# Get the index
index = pc.Index(index_name)

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_text_pymupdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def create_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


pdf_paths = [
    "Crustdata Dataset API Detailed Examples.pdf",
    "Crustdata Discovery And Enrichment API.pdf"
]

for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")
    try:
        text = extract_text_pymupdf(pdf_path)
        chunks = list(chunk_text(text))

        for i, chunk in enumerate(tqdm(chunks)):
            embedding = create_embedding(chunk)
            index.upsert(vectors=[(f"{pdf_path}-chunk-{i}", embedding, {"text": chunk, "source": pdf_path})])

        print(f"Successfully processed and uploaded {pdf_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")

print("All data processing completed!")
