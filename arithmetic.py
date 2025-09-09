import numpy as np 
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# TBC on how exactly the prompt should be structured, might be in JSON format
example_prompt = ["This is a positive prompt", "This is a negative prompt"]


def embed(text, model = "text-embedding-3-small"):
    """
    Convert text into embeddings with OpenAI's embedding model.
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding

def normalize_embedding(embedding):
    """
    Normalize the embedding to have a unit length.
    """
    return embedding / np.linalg.norm(embedding)


def preprocess_embedding(prompt, lambda_value = 0.9):
    """
    Preprocess the embedding of the prompt.
    First, embed the prompt.
    After that, we combine both the positive and negative embedding into a single embedding with the formula:
    positive_embedding * lambda_value - negative_embedding * (1 - lambda_value)
    We normalize the embedding to have a unit length.
    """
    if len(prompt) == 0:
        print("Prompt is empty")
        return None
    try:
        positive_embedding = np.array(embed(prompt[0])) if prompt[0] != "" else np.zeros(1536)
        negative_embedding = np.array(embed(prompt[1])) if prompt[1] != "" else np.zeros(1536)
        combined_embedding = normalize_embedding(positive_embedding * lambda_value - negative_embedding * (1 - lambda_value))
        # Convert back to Python list for Pinecone compatibility
        return combined_embedding.tolist()
    except Exception as e:
        print(f"Error preprocessing embedding: {e}")
        return None