import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai

# Load your existing environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 

client = genai.Client(api_key=api_key)
qdrant = QdrantClient("localhost", port=6333)

collection_name = "enterprise_knowledge"

# Wipe the old collection if it exists to avoid dimension mismatches
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)

# Recreate it locked to 3072 dimensions for the new model
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# Read and chunk the master library
with open("AI_Data Science MASTER.txt", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)] 

print(f"Starting embedding for {len(chunks)} chunks.")
print("If we hit the free-tier rate limit (100 per minute), the script will automatically pause and resume. Do not close the terminal.")

# Embed and upload using the new Gemini Embedding 2 model with Rate Limit handling
points = []
for idx, chunk in enumerate(chunks):
    success = False
    while not success:
        try:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=chunk
            )
            
            embedding = response.embeddings[0].values 
            points.append(PointStruct(id=idx, vector=embedding, payload={"text": chunk}))
            success = True
            
            # Print a progress update every 50 chunks so you know it hasn't frozen
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                print(f"⏳ Rate limit hit at chunk {idx + 1}. Sleeping for 15 seconds to let the quota cool down...")
                time.sleep(15)
            else:
                # If it's a different error, crash loudly so we can fix it
                raise e

# Push the vectors to the database
qdrant.upsert(collection_name=collection_name, points=points)

print(f"✅ Successfully embedded and ingested {len(chunks)} chunks into Qdrant!")