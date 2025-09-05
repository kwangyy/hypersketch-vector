from arithmetic import preprocess_embedding
import random
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sref-test-index")

def retrieve(query, top_k = 5, prod = True):
    """
    Given the user's positive and negative prompts, we need to retrieve the most relevant images from the database
    First, we preprocess the user's query, both positive and negative prompts to get one embedding
    Then, we call this into the vectordb to retrieve the most relevant images
    Then, we return the top 3 most relevant images, selecting one image from the top 3 most relevant images before returning it.
    """
    # Preprocess the user's query, both positive and negative prompts to get one embedding
    embedding = preprocess_embedding(query)

    # Given the user's query, we need to retrieve the most relevant images from the database
    # Call this into the vectordb
    results = index.query(vector = embedding, top_k=top_k, include_metadata=True)
    images = [result.metadata["picture_id"] for result in results.matches]
    # Randomly select one image from the top 3 most relevant images
    if prod:
        image = random.choice(images)
        # Return image
        return image
    else:
        # Retrieve the top 3 most relevant images from the database
    
        return images


