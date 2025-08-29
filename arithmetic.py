import numpy as np 
from openai import OpenAI
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

def preprocess_embedding(prompt, alpha = 0.5):
    """
    Preprocess the embedding of the prompt.
    First, embed the chunks of the prompt.
    Then, average the embeddings of the chunks themselves.
    After that, we combine both the positive and negative embedding into a single embedding with the formula:
    positive_embedding * alpha - negative_embedding * (1 - alpha)
    We normalize the embedding to have a unit length.
    """
    # TBC on how exactly the prompt should be fed in, will change the preprocessing method 
    positive_embedding = np.mean([embed(p) for p in prompt[0]], axis = 0)
    negative_embedding = np.mean([embed(p) for p in prompt[1]], axis = 0)
    return normalize_embedding(positive_embedding * alpha - negative_embedding * (1 - alpha))
