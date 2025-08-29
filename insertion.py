import weaviate 
import json 
from arithmetic import embed

# Ok we figure this out later

# But basically, given the descriptions.json, we just insert all of the data into the database itself
with open("descriptions.json", "r") as f:
    descriptions = json.load(f)

for description in descriptions:
    # Get the id, url and description from the json file
    id = description["id"]
    url = description["url"]
    description_list = description["description"]
    
    # Embed the description
    description_embedding = [embed(description) for description in description_list]

    # Feed it into the weaviate vector database

