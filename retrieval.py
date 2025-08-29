from arithmetic import preprocess_embedding
import random

images = [1,2,3]
def retrieve(query):
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

    # Retrieve the top 3 most relevant images from the database
    
    # Randomly select one image from the top 3 most relevant images
    image = random.choice(images)
    # Return image
    return image


