import numpy as np 
from openai import OpenAI
client = OpenAI()

# TBC on how exactly the prompt should be structured, might be in JSON format
example_prompt = ["This is a positive prompt", "This is a negative prompt"]

# Define aspect weights for different types of descriptions
ASPECT_WEIGHTS = {
    "subject": 2.0,      # Main subject gets highest weight
    "style": 1.8,        # Style is very important
    "material": 1.8,     # Materials are very important
    "composition": 1.5,  # Composition is moderately important
    "background": 1.0,   # Background gets standard weight
    "colors": 1.0,       # Colors get standard weight
    "mood": 1.0,         # Mood gets standard weight
    "lighting": 0.8,     # Lighting gets slightly lower weight
    "texture": 0.8,      # Texture gets slightly lower weight
    "medium": 0.7,       # Medium gets lower weight
    "time": 0.5          # Time of day gets lowest weight
}

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

def embed_with_weights(descriptions, aspect_weights=None):
    """
    Embed descriptions with different weights for different aspects.
    
    Args:
        descriptions: List of description strings
        aspect_weights: Dict mapping aspect keywords to weights
    
    Returns:
        Weighted average embedding
    """
    if aspect_weights is None:
        aspect_weights = ASPECT_WEIGHTS
    
    weighted_embeddings = []
    total_weight = 0
    
    for desc in descriptions:
        # Determine which aspect this description belongs to
        desc_lower = desc.lower()
        weight = 1.0  # default weight
        
        for aspect, aspect_weight in aspect_weights.items():
            if aspect in desc_lower:
                weight = aspect_weight
                break
        
        # Get embedding for this description
        embedding = embed(desc)
        weighted_embeddings.append(embedding * weight)
        total_weight += weight
    
    # Calculate weighted average
    if total_weight > 0:
        weighted_avg = np.sum(weighted_embeddings, axis=0) / total_weight
        return normalize_embedding(weighted_avg)
    else:
        # Fallback to simple average if no weights applied
        return normalize_embedding(np.mean([embed(desc) for desc in descriptions], axis=0))

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

def preprocess_embedding_weighted(prompt, alpha=0.5, aspect_weights=None):
    """
    Preprocess the embedding with aspect weighting applied.
    """
    # Apply weighted embedding to positive and negative prompts
    positive_embedding = embed_with_weights(prompt[0], aspect_weights)
    negative_embedding = embed_with_weights(prompt[1], aspect_weights)
    
    return normalize_embedding(positive_embedding * alpha - negative_embedding * (1 - alpha))
