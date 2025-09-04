from pinecone import Pinecone
import os
import json 
import time
from typing import List, Dict, Set
from arithmetic import embed
from dotenv import load_dotenv

load_dotenv()

index_name = "sref-test-index"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize index
if not pc.has_index(index_name):
    # Using standard create_index since we're bringing our own OpenAI embeddings
    # text-embedding-3-small has 1536 dimensions
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-3-small dimension
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        },
        deletion_protection="enabled"
    )

index = pc.Index(index_name)

def get_existing_picture_ids():
    """
    Get all existing picture IDs from Pinecone.
    Since we store vectors as {picture_id}_{description_index}, we need to extract the picture_id part.
    Returns a set of picture IDs that already exist in the database.
    """
    try:
        existing_picture_ids = set()
        
        # Use list() to get all vector IDs - it returns a generator
        list_result = index.list()
        
        # Extract picture IDs from vector IDs
        for vector_id in list_result:
            # Vector IDs are in format: picture_id_description_index
            # So we need to extract the picture_id part (everything before the last underscore)
            if '_' in vector_id:
                picture_id = '_'.join(vector_id.split('_')[:-1])  # Everything except the last part
                existing_picture_ids.add(picture_id)
            else:
                # If no underscore, it might be a direct picture_id (legacy format)
                existing_picture_ids.add(vector_id)
        
        return existing_picture_ids
        
    except Exception as e:
        print(f"Error getting existing picture IDs: {e}")
        # Fallback: try the query approach
        try:
            print("Trying fallback query approach...")
            dummy_query_vector = [0.0] * 1536
            result = index.query(
                vector=dummy_query_vector,
                top_k=10000,
                include_metadata=True
            )
            
            existing_picture_ids = set()
            for match in result.matches:
                if 'picture_id' in match.metadata:
                    existing_picture_ids.add(match.metadata['picture_id'])
            
            return existing_picture_ids
        except Exception as e2:
            print(f"Fallback query also failed: {e2}")
            return set()

def batch_embed_descriptions(descriptions_to_embed: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Batch embed descriptions to optimize API calls and respect rate limits.
    """
    embeddings = []
    
    for i in range(0, len(descriptions_to_embed), batch_size):
        batch = descriptions_to_embed[i:i + batch_size]
        
        # Add small delay to respect rate limits
        if i > 0:
            time.sleep(0.1)
        
        try:
            # Embed each description in the batch
            batch_embeddings = [embed(desc) for desc in batch]
            embeddings.extend(batch_embeddings)
            print(f"   📝 Embedded batch {i//batch_size + 1}: {len(batch)} descriptions")
            
        except Exception as e:
            print(f"❌ Failed to embed batch starting at index {i}: {e}")
            # Add None placeholders for failed embeddings
            embeddings.extend([None] * len(batch))
    
    return embeddings

def is_valid_id(picture_id) -> bool:
    """
    Check if the picture ID is valid for Pinecone (not Infinity, null, etc.)
    """
    if picture_id is None:
        return False
    if isinstance(picture_id, float) and (picture_id == float('inf') or picture_id != picture_id):  # inf or NaN
        return False
    if str(picture_id).lower() in ['infinity', 'inf', 'null', 'undefined', 'none']:
        return False
    return True

def prepare_vectors_for_upsert(new_pictures: Dict) -> List[Dict]:
    """
    Prepare all vectors for the pictures that need to be uploaded.
    Only embeds descriptions for pictures that are confirmed to be new.
    """
    all_vectors = []
    
    # Collect all descriptions that need embedding
    all_descriptions = []
    description_metadata = []
    invalid_pictures = []
    
    for filename, picture_data in new_pictures.items():
        picture_id = picture_data["id"]
        path = picture_data["path"]
        description_list = picture_data["description"]
        
        # Validate the picture ID
        if not is_valid_id(picture_id):
            invalid_pictures.append(filename)
            print(f"⚠️  Skipping {filename} - invalid ID: {picture_id}")
            continue
        
        for i, desc in enumerate(description_list):
            all_descriptions.append(desc)
            description_metadata.append({
                "picture_id": str(picture_id),  # Ensure it's a string
                "filename": filename,
                "path": path,
                "description_index": i,
                "vector_id": f"{picture_id}_{i}"
            })
    
    if invalid_pictures:
        print(f"⚠️  Found {len(invalid_pictures)} pictures with invalid IDs that will be skipped")
    
    print(f"🔤 Embedding {len(all_descriptions)} descriptions for {len(new_pictures)} new pictures...")
    
    # Batch embed all descriptions
    all_embeddings = batch_embed_descriptions(all_descriptions)
    
    # Create vectors from embeddings
    for embedding, metadata in zip(all_embeddings, description_metadata):
        if embedding is not None:  # Skip failed embeddings
            all_vectors.append({
                "id": metadata["vector_id"],
                "values": embedding,
                "metadata": {
                    "picture_id": metadata["picture_id"],
                    "filename": metadata["filename"],
                    "path": metadata["path"],
                    "description": all_descriptions[description_metadata.index(metadata)],
                    "description_index": metadata["description_index"]
                }
            })
    
    return all_vectors

def upsert_vectors_in_batches(vectors: List[Dict], batch_size: int = 100):
    """
    Upsert vectors to Pinecone in batches with proper error handling.
    """
    successful_upserts = 0
    failed_upserts = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        
        try:
            index.upsert(vectors=batch)
            successful_upserts += len(batch)
            print(f"   ✅ Uploaded batch {i//batch_size + 1}: {len(batch)} vectors")
            
            # Small delay between batches
            time.sleep(0.1)
            
        except Exception as e:
            failed_upserts += len(batch)
            print(f"   ❌ Failed to upload batch {i//batch_size + 1}: {e}")
    
    return successful_upserts, failed_upserts

def main():
    """
    Main function to process and upload picture descriptions to Pinecone.
    """
    # Load descriptions
    print("📂 Loading descriptions...")
    with open("descriptions/descriptions.json", "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    print(f"📊 Found {len(descriptions)} pictures in descriptions file")

    # STEP 1: Check for duplicates BEFORE any embedding
    print("\n🔍 Checking for existing pictures in database...")
    existing_picture_ids = get_existing_picture_ids()

    print(f"Found {len(existing_picture_ids)} pictures already in database")
    
    # Count how many of our current pictures are new
    all_picture_ids = [str(descriptions[filename]["id"]) for filename in descriptions if is_valid_id(descriptions[filename]["id"])]
    new_picture_count = len([pid for pid in all_picture_ids if pid not in existing_picture_ids])
    
    print(f"Need to process {new_picture_count} new pictures")

    # STEP 2: Filter out existing pictures (NO EMBEDDING YET!)
    new_pictures = {}
    skipped_count = 0
    invalid_count = 0

    for filename in descriptions:
        picture_id = descriptions[filename]["id"]
        
        # Check for invalid IDs first
        if not is_valid_id(picture_id):
            print(f"⚠️  Skipping {filename} - invalid ID: {picture_id}")
            invalid_count += 1
            continue
        
        if str(picture_id) in existing_picture_ids:
            print(f"⏭️  Skipping {filename} (ID: {picture_id}) - already exists in database")
            skipped_count += 1
        else:
            new_pictures[filename] = descriptions[filename]
            print(f"📋 Queued {filename} (ID: {picture_id})")

    # STEP 3: Only NOW embed and upload new pictures
    if new_pictures:
        print(f"\n🚀 Processing {len(new_pictures)} new pictures...")
        
        # Prepare all vectors (this is where embedding happens)
        vectors_to_upsert = prepare_vectors_for_upsert(new_pictures)
        
        if vectors_to_upsert:
            print(f"\n📤 Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
            successful, failed = upsert_vectors_in_batches(vectors_to_upsert)
            
            print(f"\n📊 Upload Summary:")
            print(f"   • New pictures processed: {len(new_pictures)}")
            print(f"   • Vectors uploaded successfully: {successful}")
            print(f"   • Vectors failed: {failed}")
            print(f"   • Pictures skipped (duplicates): {skipped_count}")
            print(f"   • Pictures skipped (invalid IDs): {invalid_count}")
            print(f"   • Total pictures: {len(descriptions)}")
            print(f"   • API calls saved by duplicate checking: ~{skipped_count * 20}")  # Assuming ~20 descriptions per picture
        else:
            print("❌ No vectors to upload due to embedding failures")
    else:
        print(f"\n✅ All pictures already exist in database!")
        print(f"📊 Final Summary:")
        print(f"   • Pictures skipped (duplicates): {skipped_count}")
        print(f"   • Pictures skipped (invalid IDs): {invalid_count}")
        print(f"   • Total pictures: {len(descriptions)}")
        print(f"   • API calls saved: ~{skipped_count * 20}")  # Assuming ~20 descriptions per picture


if __name__ == "__main__":
    main()
