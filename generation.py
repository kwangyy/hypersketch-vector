from openai import OpenAI
from pydantic import BaseModel
import json 
client = OpenAI()

# To be changed to the actual pictures
pictures = [1,2,3,4,5,6,7,8,9,10]
descriptions = {}

prompt = """
You are a helpful assistant that will generate different descriptions given a picture.
You will be given a picture and you will need to generate different descriptions for it. 
The number of descriptions you will generate will be based on the different aspects of the picture.
The different aspects of the picture that you will need to take note are:
- The main subject of the picture
- The background of the picture
- The colors of the picture
- The mood of the picture
- The time of day of the picture
- The style of the picture (if there are any)
- The medium used to create the picture
- The composition of the picture
- The lighting of the picture
- The texture of the picture

You are to return a JSON object in the following format:
    {
        "id": 1,
        "url": "https://example.com/image.jpg",
        "description": ["Description 1 of the picture", "Description 2 of the picture", "Description 3 of the picture"]
    }
"""
class Description(BaseModel):
    """
    Return the description of the picture in JSON format.
    id: id of the picture
    url: url of the picture
    description: list of descriptions of the picture
    """
    id: int
    url: str
    description: list[str]

for picture in pictures:
    response = client.chat.completions.create(
    model="gpt-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
        ],
    }],
    response_format = Description,
    )

    description_json = response.choices[0].message.content
    descriptions[picture] = description_json

with open("descriptions.json", "w") as f:
    json.dump(descriptions, f)