from openai import OpenAI
from pydantic import BaseModel
import json 
import os 
from pathlib import Path
from dotenv import load_dotenv
import logging
import threading
import concurrent.futures
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
    "descriptions_dir": "descriptions",
    "model": "gpt-4", 
    "max_retries": 3,
    "batch_size": 10  # Process in batches for better progress tracking
}

client = OpenAI()

def load_existing_evaluations(descriptions_dir: str = CONFIG["descriptions_dir"]) -> Dict:
    """Load existing evaluations from the main evaluation.json file."""
    existing_evaluations = {}
    descriptions_path = Path(descriptions_dir)
    eval_file = descriptions_path / "evaluation.json"
    
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                existing_evaluations = json.load(f)
            logger.info(f"Loaded {len(existing_evaluations)} evaluations from {eval_file.name}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load {eval_file.name}: {e}")
    else:
        logger.info("No existing evaluation.json found - starting fresh")
    
    logger.info(f"Total existing evaluations: {len(existing_evaluations)}")
    return existing_evaluations

def load_descriptions_and_picture_ids(descriptions_dir: str = CONFIG["descriptions_dir"]) -> tuple[Dict, Dict]:
    """Load descriptions and picture_id mappings."""
    descriptions_path = Path(descriptions_dir)
    
    # Load descriptions.json
    descriptions_file = descriptions_path / "descriptions.json"
    descriptions = {}
    if descriptions_file.exists():
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                descriptions = json.load(f)
            logger.info(f"Loaded {len(descriptions)} descriptions")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Could not load descriptions.json: {e}")
            return {}, {}
    
    # Load picture_id.json
    picture_id_file = descriptions_path / "picture_id.json"
    picture_ids = {}
    if picture_id_file.exists():
        try:
            with open(picture_id_file, "r", encoding="utf-8") as f:
                picture_ids = json.load(f)
            logger.info(f"Loaded {len(picture_ids)} picture ID mappings")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Could not load picture_id.json: {e}")
            return descriptions, {}
    
    return descriptions, picture_ids

def create_positive_paraphrase(description_list: List[str]) -> str:
    """Create a positive paraphrase that emphasizes the good qualities."""
    description_text = "\n".join([f"- {desc}" for desc in description_list])
    
    prompt = f"""
You are a text rewriter. Rewrite image descriptors to sound positive and appealing.

Input descriptors:
{description_text}

Task: Pick the 5 most important descriptors and rewrite them to emphasize good qualities.

Do not format as JSON. Do not add quotes. Do not add numbers. Just write the descriptors separated by commas.

Example output: Beautiful elegant design, Masterfully crafted architecture, Stunning aerodynamic curves, Sophisticated minimalist aesthetic, Expertly refined visual language
"""
    
    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def create_negative_paraphrase(description_list: List[str]) -> str:
    """Create a negative paraphrase that highlights potential issues."""
    description_text = "\n".join([f"- {desc}" for desc in description_list])
    
    prompt = f"""
You are a text rewriter. Rewrite image descriptors to sound critical and negative.

Input descriptors:
{description_text}

Task: Pick the 5 most important descriptors and rewrite them to emphasize flaws and problems.

Do not format as JSON. Do not add quotes. Do not add numbers. Just write the descriptors separated by commas.

Example output: Bland monochrome lacking color variety, Overly technical sterile sketch style, Overly smooth forms lacking texture detail, Cold minimalism devoid of character, Impersonal futuristic detachment
"""
    
    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def create_noisy_paraphrase(description_list: List[str]) -> str:
    """Create a noisy paraphrase with random variations and distractions."""
    description_text = "\n".join([f"- {desc}" for desc in description_list])
    
    prompt = f"""
You are a text rewriter. Rewrite image descriptors with creative variations and random elements.

Input descriptors:
{description_text}

Task: Pick the 5 most important descriptors and rewrite them with creative interpretations and tangential elements.

Do not format as JSON. Do not add quotes. Do not add numbers. Just write the descriptors separated by commas.

Example output: Chaotic black-and-white scattered linework, Jumbled architectural doodle-like style, Weirdly smooth blob-like organic shapes, Confusingly minimal yet somehow busy futuristic mess, Mixed contemporary visual chaos
"""
    
    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def generate_all_paraphrases(description_list: List[str]) -> Dict[str, str]:
    """Generate all three types of paraphrases concurrently using threading."""
    paraphrases = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three tasks
        future_positive = executor.submit(create_positive_paraphrase, description_list)
        future_negative = executor.submit(create_negative_paraphrase, description_list)
        future_noisy = executor.submit(create_noisy_paraphrase, description_list)
        
        # Collect results
        try:
            paraphrases["positive"] = future_positive.result()
            paraphrases["negative"] = future_negative.result()
            paraphrases["noisy"] = future_noisy.result()
        except Exception as e:
            logger.error(f"Error in paraphrase generation: {e}")
            raise
    
    return paraphrases

class EvaluationEntry(BaseModel):
    """
    Return the evaluation entry in JSON format.
    paraphrased_prompt: the paraphrased version of the description
    paraphrase_type: type of paraphrase (positive, negative, or noisy)
    picture_id: the ID from picture_id.json that maps to the image filename
    """
    paraphrased_prompt: str
    paraphrase_type: str
    picture_id: str

def process_single_description(filename: str, description_data: Dict, picture_ids: Dict, existing_evaluations: Dict) -> Optional[List[Dict]]:
    """Process a single description to create an evaluation entry."""
    
    # Skip if already processed
    if filename in existing_evaluations:
        logger.debug(f"Skipping {filename} - already processed")
        return None
    
    # Get the picture ID from the description data
    picture_id = description_data.get("id")
    if not picture_id:
        logger.warning(f"No ID found in description data for {filename}")
        return None
    
    # Verify the picture ID exists in picture_id.json
    if picture_id not in picture_ids:
        logger.warning(f"Picture ID {picture_id} not found in picture_id.json for {filename}")
        return None
    
    description_list = description_data.get("description", [])
    if not description_list:
        logger.warning(f"No description list found for {filename}")
        return None
    
    logger.info(f"Processing evaluation for: {filename}")
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            # Generate all three paraphrases concurrently
            paraphrases = generate_all_paraphrases(description_list)
            
            # Split each paraphrase by commas and create individual data points
            evaluation_entries = []
            
            # Process positive paraphrases
            positive_descriptors = [desc.strip() for desc in paraphrases["positive"].split(",") if desc.strip()]
            for descriptor in positive_descriptors:
                evaluation_entries.append({
                    "paraphrased_prompt": descriptor,
                    "paraphrase_type": "positive",
                    "picture_id": picture_id
                })
            
            # Process negative paraphrases
            negative_descriptors = [desc.strip() for desc in paraphrases["negative"].split(",") if desc.strip()]
            for descriptor in negative_descriptors:
                evaluation_entries.append({
                    "paraphrased_prompt": descriptor,
                    "paraphrase_type": "negative",
                    "picture_id": picture_id
                })
            
            # Process noisy paraphrases
            noisy_descriptors = [desc.strip() for desc in paraphrases["noisy"].split(",") if desc.strip()]
            for descriptor in noisy_descriptors:
                evaluation_entries.append({
                    "paraphrased_prompt": descriptor,
                    "paraphrase_type": "noisy",
                    "picture_id": picture_id
                })
            
            logger.info(f"Successfully processed 3 evaluation entries for: {filename}")
            return evaluation_entries
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{CONFIG['max_retries']} failed for {filename}: {e}")
            if attempt == CONFIG["max_retries"] - 1:
                logger.error(f"Failed to process {filename} after {CONFIG['max_retries']} attempts")
                return [{
                    "error": str(e),
                    "filename": filename,
                    "picture_id": picture_id
                }]
    
    return None

def save_evaluations(new_evaluations: Dict, descriptions_dir: str = CONFIG["descriptions_dir"]) -> Optional[str]:
    """Save new evaluations by merging them into the main evaluation.json file."""
    if not new_evaluations:
        logger.info("No new evaluations to save.")
        return None
    
    # Create descriptions folder if it doesn't exist
    os.makedirs(descriptions_dir, exist_ok=True)
    
    # Path to the main evaluation file
    main_eval_file = f"{descriptions_dir}/evaluation.json"
    
    try:
        # Load existing evaluations from the main file
        existing_evaluations = {}
        if os.path.exists(main_eval_file):
            try:
                with open(main_eval_file, "r", encoding="utf-8") as f:
                    existing_evaluations = json.load(f)
                logger.info(f"Loaded {len(existing_evaluations)} existing evaluations from {main_eval_file}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not load existing evaluations: {e}. Starting fresh.")
                existing_evaluations = {}
        
        # Merge new evaluations with existing ones
        merged_evaluations = existing_evaluations.copy()
        merged_evaluations.update(new_evaluations)
        
        # Save the merged evaluations back to the main file
        with open(main_eval_file, "w", encoding='utf-8') as f:
            json.dump(merged_evaluations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully merged {len(new_evaluations)} new evaluations into {main_eval_file}")
        logger.info(f"Total evaluations in file: {len(merged_evaluations)}")
        return main_eval_file
        
    except Exception as e:
        logger.error(f"Failed to save evaluations: {e}")
        raise

def main(descriptions_dir: str = CONFIG["descriptions_dir"]) -> Optional[str]:
    """Main processing function."""
    logger.info("Starting evaluation generation")
    
    # Load existing evaluations
    existing_evaluations = load_existing_evaluations(descriptions_dir)
    
    # Load descriptions and picture IDs
    descriptions, picture_ids = load_descriptions_and_picture_ids(descriptions_dir)
    
    if not descriptions:
        logger.error("No descriptions found. Please run generation.py first.")
        return None
    
    if not picture_ids:
        logger.error("No picture ID mappings found.")
        return None
    
    # Filter out already processed items
    new_items = {filename: data for filename, data in descriptions.items() 
                if filename not in existing_evaluations}
    
    if not new_items:
        logger.info("No new descriptions to process. All items have already been evaluated.")
        return None
    
    # Process descriptions
    new_evaluations = {}
    successful_count = 0
    error_count = 0
    
    for idx, (filename, description_data) in enumerate(new_items.items()):
        result = process_single_description(filename, description_data, picture_ids, existing_evaluations)
        
        if result:
            new_evaluations[filename] = result
            
            if any("error" in entry for entry in result):
                error_count += 1
            else:
                successful_count += 1
        
        # Progress update
        current_progress = idx + 1
        if current_progress % CONFIG["batch_size"] == 0 or current_progress == len(new_items):
            total_data_points = successful_count * 3
            logger.info(f"Progress: {current_progress}/{len(new_items)} images processed "
                       f"({total_data_points} data points created, {error_count} errors)")
    
    # Save new evaluations by merging with existing ones
    output_file = save_evaluations(new_evaluations, descriptions_dir)
    
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
