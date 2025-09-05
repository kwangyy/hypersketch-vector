from openai import OpenAI
from pydantic import BaseModel
import json 
import os 
import base64
from pathlib import Path
from dotenv import load_dotenv
import uuid
import logging
from typing import Dict, List, Optional
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "image_folder": "Kwang Yang SREF Tests",
    "descriptions_dir": "descriptions",
    "model": "gpt-5", 
    "image_extensions": {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'},
    "max_retries": 3,
    "batch_size": 10  # Process in batches for better progress tracking
}

client = OpenAI()

def load_existing_descriptions(descriptions_dir: str = CONFIG["descriptions_dir"]) -> Dict:
    """Load existing descriptions from the main descriptions.json file."""
    existing_descriptions = {}
    descriptions_path = Path(descriptions_dir)
    main_file = descriptions_path / "descriptions.json"
    
    if main_file.exists():
        try:
            with open(main_file, "r", encoding="utf-8") as f:
                existing_descriptions = json.load(f)
            logger.info(f"Loaded {len(existing_descriptions)} descriptions from {main_file.name}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load {main_file.name}: {e}")
    else:
        logger.info("No existing descriptions.json found - starting fresh")
    
    logger.info(f"Total existing descriptions: {len(existing_descriptions)}")
    return existing_descriptions

def get_image_files(image_folder: str = CONFIG["image_folder"], 
                   existing_descriptions: Optional[Dict] = None) -> tuple[List[Path], List[Path]]:
    """Get all image files and filter out already processed ones."""
    if existing_descriptions is None:
        existing_descriptions = {}
    
    image_extensions = CONFIG["image_extensions"]
    all_image_files = []
    
    image_path = Path(image_folder)
    if not image_path.exists():
        logger.error(f"Image folder does not exist: {image_folder}")
        return [], []
    
    for file_path in image_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            all_image_files.append(file_path)
    
    # Only process images that haven't been processed before
    new_image_files = [img for img in all_image_files if img.name not in existing_descriptions]
    
    logger.info(f"Found {len(all_image_files)} total image files")
    logger.info(f"Found {len(new_image_files)} new image files to process")
    
    return all_image_files, new_image_files

def encode_image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 encoded string with error handling."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise

def create_prompt(image_path: Path) -> str:
    """Create the prompt with the specific image path"""
    # Generate UUID as a string for the prompt
    generated_uuid = str(uuid.uuid4())
    
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
        "id": "{generated_uuid}",
        "path": "{image_path}",
        "description": [
            "Descriptor 1 of the picture",
            "Descriptor 2 of the picture",
            "Descriptor 3 of the picture"
        ]
    }}
The id and path have already been provided to you in the example itself, so reuse these exact values.
"""

class Description(BaseModel):
    """
    Return the description of the picture in JSON format.
    id: id of the picture (UUID string)
    path: path of the picture
    description: list of descriptions of the picture
    """
    id: str  # Changed from int to str to handle UUIDs
    path: str
    description: list[str]

def process_single_image(image_path: Path, idx: int) -> Dict:
    """Process a single image and return its description."""
    logger.info(f"Processing image {idx}: {image_path.name}")
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            # Encode the image to base64
            base64_image = encode_image_to_base64(image_path)
            
            response = client.chat.completions.create(
                model=CONFIG["model"],
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
            
            # Validate that we got expected fields
            if not all(key in description_data for key in ["id", "path", "description"]):
                raise ValueError(f"Missing required fields in response: {description_data}")
            
            # Validate that the ID is a proper string (not Infinity, null, etc.)
            if not isinstance(description_data["id"], str) or not description_data["id"].strip():
                raise ValueError(f"Invalid ID in response: {description_data['id']}")
            
            # Additional validation - ensure description is a list
            if not isinstance(description_data["description"], list):
                raise ValueError(f"Description must be a list, got: {type(description_data['description'])}")
            
            logger.info(f"Successfully processed: {image_path.name}")
            return description_data
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{CONFIG['max_retries']} failed for {image_path.name}: {e}")
            if attempt == CONFIG["max_retries"] - 1:
                # Final attempt failed, store error information
                logger.error(f"Failed to process {image_path.name} after {CONFIG['max_retries']} attempts")
                return {
                    "error": str(e),
                    "path": str(image_path),
                    "id": idx
                }
    
    # This shouldn't be reached, but just in case
    return {
        "error": "Unknown error",
        "path": str(image_path),
        "id": idx
    }

def save_descriptions(new_descriptions: Dict, descriptions_dir: str = CONFIG["descriptions_dir"]) -> Optional[str]:
    """Save new descriptions by merging them into the main descriptions.json file."""
    if not new_descriptions:
        logger.info("No new descriptions to save.")
        return None
    
    # Create descriptions folder if it doesn't exist
    os.makedirs(descriptions_dir, exist_ok=True)
    
    # Path to the main descriptions file
    main_descriptions_file = f"{descriptions_dir}/descriptions.json"
    
    try:
        # Load existing descriptions from the main file
        existing_descriptions = {}
        if os.path.exists(main_descriptions_file):
            try:
                with open(main_descriptions_file, "r", encoding="utf-8") as f:
                    existing_descriptions = json.load(f)
                logger.info(f"Loaded {len(existing_descriptions)} existing descriptions from {main_descriptions_file}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not load existing descriptions: {e}. Starting fresh.")
                existing_descriptions = {}
        
        # Merge new descriptions with existing ones
        merged_descriptions = existing_descriptions.copy()
        merged_descriptions.update(new_descriptions)
        
        # Save the merged descriptions back to the main file
        with open(main_descriptions_file, "w", encoding='utf-8') as f:
            json.dump(merged_descriptions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully merged {len(new_descriptions)} new descriptions into {main_descriptions_file}")
        logger.info(f"Total descriptions in file: {len(merged_descriptions)}")
        return main_descriptions_file
        
    except Exception as e:
        logger.error(f"Failed to save descriptions: {e}")
        raise

def main(image_folder: str = CONFIG["image_folder"], 
         descriptions_dir: str = CONFIG["descriptions_dir"]) -> Optional[str]:
    """Main processing function."""
    logger.info("Starting image description generation")
    
    # Load existing descriptions
    existing_descriptions = load_existing_descriptions(descriptions_dir)
    
    # Get image files to process
    all_image_files, new_image_files = get_image_files(image_folder, existing_descriptions)
    
    if not new_image_files:
        logger.info("No new images to process. All images have already been described.")
        return None
    
    # Process images
    new_descriptions = {}
    successful_count = 0
    error_count = 0
    
    for idx, image_path in enumerate(new_image_files, 1):
        result = process_single_image(image_path, idx)
        new_descriptions[image_path.name] = result
        
        if "error" in result:
            error_count += 1
        else:
            successful_count += 1
        
        # Progress update
        if idx % CONFIG["batch_size"] == 0 or idx == len(new_image_files):
            logger.info(f"Progress: {idx}/{len(new_image_files)} images processed "
                       f"({successful_count} successful, {error_count} errors)")
    
    # Save new descriptions by merging with existing ones
    output_file = save_descriptions(new_descriptions, descriptions_dir)
    
    logger.info(f"Processing complete! {successful_count} successful, {error_count} errors")
    return output_file

if __name__ == "__main__":
    try:
        output_file = main()
        if output_file:
            logger.info(f"Results saved to: {output_file}")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise