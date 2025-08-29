from openai import OpenAI
from pydantic import BaseModel
import json 
import os 
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# Define the folder containing the images
image_folder = "Kwang Yang SREF Tests"

# Get all image files from the folder
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
image_files = []

# Get the first 5 image files from the folder
for file_path in Path(image_folder).iterdir():
    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
        image_files.append(file_path)
        if len(image_files) >= 5:  # Stop after finding 5 images
            break

print(f"Found {len(image_files)} image files to process")

descriptions = {}

def encode_image_to_base64(image_path):
    """Convert image file to base64 encoded string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_prompt(image_path):
    """Create the prompt with the specific image path"""
    return f"""
You are a helpful assistant that will generate different descriptions given a picture.
You will be given a picture and you will need to generate a list of descriptors for it. 
You are to focus on the descriptors of the pictures themselves. The object identity is not as important.
The number of descriptors you will generate will be based on the different aspects of the picture.

The different aspects of the picture that you will need to take note are, in order of importance:
- The colours of the picture (foreground objects and background colours must be emphasized)
- The style of the picture (e.g. photo, painting, sketch, rendering, abstract)
- The material and shape properties of the picture (e.g. metal, chrome, plastic, wood, glossy, sharp, curvy, sleek, fragmented)
- The style of objects (e.g. futuristic, rustic, minimalistic)
- The main subject of the picture (the object itself, e.g. tree, spaceship, person)
- The background of the picture
- The mood of the picture
- The time of day of the picture
- The medium used to create the picture
- The composition of the picture
- The lighting of the picture
- The texture of the picture

If the aspects of the picture are not clear or if any aspect of the picture is indeterminate, you should omit that aspect from the description.
Do NOT make up any new information at all. Only use what is given to you in the picture. 
When you return the list, you are to NOT have the sub-topic in front of the descriptor.

You are to return a JSON object in the following format:
    {{
        "id": 1,
        "path": "{image_path}",
        "description": [
            "Descriptor 1 of the picture",
            "Descriptor 2 of the picture",
            "Descriptor 3 of the picture"
        ]
    }}
The path has already been provided to you in the example itself, so reuse this path.
"""

class Description(BaseModel):
    """
    Return the description of the picture in JSON format.
    id: id of the picture
    path: path of the picture
    description: list of descriptions of the picture
    """
    id: int
    path: str
    description: list[str]

# Process each image file
for idx, image_path in enumerate(image_files, 1):
    print(f"Processing image {idx}/{len(image_files)}: {image_path.name}")
    
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": create_prompt(image_path)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }],
            response_format={"type": "json_object"},
        )

        # Parse the JSON response
        response_content = response.choices[0].message.content
        description_data = json.loads(response_content)
        
        # Store the description with the filename as key
        descriptions[image_path.name] = description_data
        
        print(f"Successfully processed: {image_path.name}")
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {str(e)}")
        # Store error information
        descriptions[image_path.name] = {
            "error": str(e),
            "path": str(image_path),
            "id": idx
        }

# Create descriptions folder if it doesn't exist
os.makedirs("descriptions", exist_ok=True)

# Count existing description files
try:
    num_descriptions = len([f for f in os.listdir("descriptions") if os.path.isfile(os.path.join("descriptions", f))])
except FileNotFoundError:
    num_descriptions = 0

# Save all descriptions to JSON file
with open(f"descriptions/descriptions_{num_descriptions + 1}.json", "w", encoding='utf-8') as f:
    json.dump(descriptions, f, indent=2, ensure_ascii=False)

print(f"Processing complete! Results saved to descriptions/descriptions_{num_descriptions + 1}.json")