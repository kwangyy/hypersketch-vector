import json 
import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retrieval'))
from arithmetic import preprocess_embedding, embed
from retrieval import retrieve
import random
import numpy as np

load_dotenv()

def load_evaluation_data(file_path: str = "data/descriptions/evaluation.json"):
    """Load evaluation data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_picture_id(data):
    """Group evaluation entries by picture_id."""
    grouped = {}
    for filename, entries in data.items():
        if isinstance(entries, list):
            for entry in entries:
                picture_id = entry.get("picture_id")
                if picture_id:
                    if picture_id not in grouped:
                        grouped[picture_id] = {"positive": [], "negative": [], "noisy": []}
                    
                    paraphrase_type = entry.get("paraphrase_type")
                    if paraphrase_type in grouped[picture_id]:
                        grouped[picture_id][paraphrase_type].append(entry["paraphrased_prompt"])
    return grouped

def create_training_pairs(grouped_data, pairs_per_picture: int = 6):
    """
    Create training pairs from grouped evaluation data.
    
    Args:
        grouped_data: Dictionary with picture_id as key, containing positive/negative/noisy lists
        pairs_per_picture: Number of pairs to create per picture (default 6)
                          First half will be positive+negative pairs
                          Second half will be noisy+negative pairs
    
    Returns:
        List of training pairs: [{"positive": str, "negative": str, "picture_id": str, "pair_type": str}]
    """
    training_pairs = []
    
    for picture_id, prompts in grouped_data.items():
        positive_prompts = prompts.get("positive", [])
        negative_prompts = prompts.get("negative", [])
        noisy_prompts = prompts.get("noisy", [])
        
        # Skip if we don't have enough data
        if not positive_prompts or not negative_prompts:
            print(f"Warning: Insufficient data for picture_id {picture_id}")
            continue
        
        # Calculate how many of each type of pair to create
        positive_negative_pairs = pairs_per_picture // 2
        noisy_negative_pairs = pairs_per_picture - positive_negative_pairs
        
        # Create positive + negative pairs
        for i in range(positive_negative_pairs):
            positive_prompt = random.choice(positive_prompts)
            negative_prompt = random.choice(negative_prompts)
            
            training_pairs.append({
                "positive": positive_prompt,
                "negative": negative_prompt,
                "picture_id": picture_id,
                "pair_type": "positive_negative"
            })
        
        # Create noisy + negative pairs (if we have noisy prompts)
        if noisy_prompts:
            for i in range(noisy_negative_pairs):
                noisy_prompt = random.choice(noisy_prompts)
                negative_prompt = random.choice(negative_prompts)
                
                training_pairs.append({
                    "positive": noisy_prompt,
                    "negative": negative_prompt,
                    "picture_id": picture_id,
                    "pair_type": "noisy_negative"
                })
        else:
            # If no noisy prompts, create more positive+negative pairs
            for i in range(noisy_negative_pairs):
                positive_prompt = random.choice(positive_prompts)
                negative_prompt = random.choice(negative_prompts)
                
                training_pairs.append({
                    "positive": positive_prompt,
                    "negative": negative_prompt,
                    "picture_id": picture_id,
                    "pair_type": "positive_negative_fallback"
                })
    
    return training_pairs

def create_query_groups(training_pairs):
    """Group training pairs into lists of 2 strings [positive, negative] for querying."""
    query_groups = []
    
    for pair in training_pairs:
        positive = pair["positive"]
        negative = pair["negative"]
        picture_id = pair["picture_id"]
        pair_type = pair["pair_type"]
        
        # Create a group with the positive and negative prompts
        query_group = {
            "query_strings": [positive, negative],
            "picture_id": picture_id,
            "pair_type": pair_type,
            "positive": positive,
            "negative": negative
        }
        query_groups.append(query_group)
    
    return query_groups

def query_training_pairs(query_groups, prod=False):
    """Query each group using the retrieve function."""
    results = []
    
    for group in query_groups:
        query_strings = group["query_strings"]
        
        # Call retrieve with the query strings (it will handle preprocessing)
        retrieved_images = retrieve(query_strings, top_k = 10, prod=prod)
        
        # Add the results to our group
        result = {
            "original_picture_id": group["picture_id"],
            "positive": group["positive"],
            "negative": group["negative"],
            "pair_type": group["pair_type"],
            "retrieved_images": retrieved_images,
            "query_strings": query_strings
        }
        
        results.append(result)
    
    return results

def calculate_accuracy(results):
    """Calculate accuracy based on whether original_picture_id appears in retrieved_images."""
    total_queries = len(results)
    correct_retrievals = 0
    
    accuracy_by_type = {"positive_negative": {"correct": 0, "total": 0}, 
                       "noisy_negative": {"correct": 0, "total": 0}}
    
    for result in results:
        original_id = result["original_picture_id"]
        retrieved_ids = result["retrieved_images"]
        pair_type = result["pair_type"]
        
        # Check if original picture ID is in retrieved images
        is_correct = original_id in retrieved_ids
        
        if is_correct:
            correct_retrievals += 1
        
        # Track by pair type
        if pair_type in accuracy_by_type:
            accuracy_by_type[pair_type]["total"] += 1
            if is_correct:
                accuracy_by_type[pair_type]["correct"] += 1
        
        # Add accuracy flag to result
        result["is_correct"] = is_correct
    
    overall_accuracy = correct_retrievals / total_queries if total_queries > 0 else 0
    
    # Calculate accuracy by type
    for pair_type in accuracy_by_type:
        total = accuracy_by_type[pair_type]["total"]
        correct = accuracy_by_type[pair_type]["correct"]
        accuracy_by_type[pair_type]["accuracy"] = correct / total if total > 0 else 0
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_queries": total_queries,
        "correct_retrievals": correct_retrievals,
        "accuracy_by_type": accuracy_by_type
    }

def query_training_pairs_with_lambda(query_groups, lambda_value, prod=False):
    """Query each group using the retrieve function with specified lambda value."""
    # Setup Pinecone connection
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sref-test-index")
    
    results = []
    
    for group in query_groups:
        query_strings = group["query_strings"]
        
        # Use preprocess_embedding with custom lambda value
        embedding = preprocess_embedding(query_strings, lambda_value=lambda_value)
        
        # Query Pinecone directly
        pc_results = index.query(vector=embedding, top_k=10, include_metadata=True)
        retrieved_images = [result.metadata["picture_id"] for result in pc_results.matches]
        
        # Add the results to our group
        result = {
            "original_picture_id": group["picture_id"],
            "positive": group["positive"],
            "negative": group["negative"],
            "pair_type": group["pair_type"],
            "retrieved_images": retrieved_images,
            "query_strings": query_strings,
            "lambda_value": lambda_value
        }
        
        results.append(result)
    
    return results

def lambda_sweep_evaluation(query_groups, lambda_values=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
    """Test different lambda values and return results as pandas DataFrame."""
    all_results = []
    
    print("Starting lambda sweep evaluation...")
    
    for lambda_val in lambda_values:
        print(f"Testing lambda = {lambda_val:.1f}")
        
        # Get results for this lambda value
        results = query_training_pairs_with_lambda(query_groups, lambda_val, prod=False)
        accuracy_stats = calculate_accuracy(results)
        
        # Add lambda-specific info to each result
        for result in results:
            result["lambda_value"] = lambda_val
            result["is_correct"] = result["original_picture_id"] in result["retrieved_images"]
            if result["is_correct"]:
                result["rank"] = result["retrieved_images"].index(result["original_picture_id"]) + 1
            else:
                result["rank"] = None
        
        all_results.extend(results)
        
        print(f"  Lambda {lambda_val:.1f}: {accuracy_stats['overall_accuracy']:.2%} accuracy")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df

def analyze_lambda_sweep_results(df):
    """Analyze lambda sweep results and return summary DataFrames."""
    
    # 1. Overall accuracy by lambda
    accuracy_by_lambda = df.groupby('lambda_value').agg({
        'is_correct': ['mean', 'count', 'sum'],
        'rank': 'mean'
    }).round(4)
    
    accuracy_by_lambda.columns = ['accuracy', 'total_queries', 'correct_queries', 'avg_rank']
    
    # 2. Accuracy by lambda and pair type
    accuracy_by_lambda_type = df.groupby(['lambda_value', 'pair_type']).agg({
        'is_correct': ['mean', 'count', 'sum']
    }).round(4)
    
    accuracy_by_lambda_type.columns = ['accuracy', 'total_queries', 'correct_queries']
    
    # 3. Rank distribution by lambda
    rank_distribution = df[df['is_correct'] == True].groupby(['lambda_value', 'rank']).size().unstack(fill_value=0)
    
    return {
        'accuracy_by_lambda': accuracy_by_lambda,
        'accuracy_by_lambda_type': accuracy_by_lambda_type,
        'rank_distribution': rank_distribution,
        'full_results': df
    }

def save_eval_results(results, accuracy_stats, output_file: str = "data/descriptions/eval_results.json"):
    """Save evaluation results to JSON file in descriptions folder."""
    eval_data = {
        "accuracy_stats": accuracy_stats,
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results to {output_file}")
    print(f"Overall Accuracy: {accuracy_stats['overall_accuracy']:.2%}")
    print(f"Correct: {accuracy_stats['correct_retrievals']}/{accuracy_stats['total_queries']}")

def save_lambda_sweep_results(analysis_results, output_dir: str = "data/results"):
    """Save lambda sweep results as CSV files and summary."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrames as CSVs
    analysis_results['accuracy_by_lambda'].to_csv(f"{output_dir}/lambda_sweep_accuracy.csv")
    analysis_results['accuracy_by_lambda_type'].to_csv(f"{output_dir}/lambda_sweep_by_type.csv")
    analysis_results['rank_distribution'].to_csv(f"{output_dir}/lambda_sweep_ranks.csv")
    analysis_results['full_results'].to_csv(f"{output_dir}/lambda_sweep_full_results.csv", index=False)
    
    print(f"Saved lambda sweep results to {output_dir}/")
    print("\nFiles created:")
    print(f"  - lambda_sweep_accuracy.csv")
    print(f"  - lambda_sweep_by_type.csv") 
    print(f"  - lambda_sweep_ranks.csv")
    print(f"  - lambda_sweep_full_results.csv")

def main():
    """Main function to generate training pairs, query them, and evaluate results."""
    # Load evaluation data
    data = load_evaluation_data()
    
    # Group by picture_id
    grouped_data = group_by_picture_id(data)
    print(f"Found {len(grouped_data)} unique pictures")
    
    # Create training pairs (6 per picture by default, easily configurable)
    training_pairs = create_training_pairs(grouped_data, pairs_per_picture=6)
    print(f"Generated {len(training_pairs)} total training pairs")
    
    # Create query groups (lists of 2 strings each)
    query_groups = create_query_groups(training_pairs)
    print(f"Created {len(query_groups)} query groups")
    
    # Query each group using the retrieve function
    print("Querying training pairs...")
    query_results = query_training_pairs(query_groups, prod=False)  # Set prod=False to get all 3 results
    print(f"Completed {len(query_results)} queries")
    
    # Calculate accuracy
    print("\nCalculating accuracy...")
    accuracy_stats = calculate_accuracy(query_results)
    
    # Print accuracy results
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Overall Accuracy: {accuracy_stats['overall_accuracy']:.2%} ({accuracy_stats['correct_retrievals']}/{accuracy_stats['total_queries']})")
    
    for pair_type, stats in accuracy_stats['accuracy_by_type'].items():
        print(f"{pair_type.replace('_', ' ').title()} Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save evaluation results
    save_eval_results(query_results, accuracy_stats)
    
    return query_results, accuracy_stats

def main_lambda_sweep():
    """Main function for lambda sweep evaluation with pandas DataFrames."""
    # Load evaluation data
    data = load_evaluation_data()
    
    # Group by picture_id
    grouped_data = group_by_picture_id(data)
    print(f"Found {len(grouped_data)} unique pictures")
    
    # Create training pairs (6 per picture by default, easily configurable)
    training_pairs = create_training_pairs(grouped_data, pairs_per_picture=6)
    print(f"Generated {len(training_pairs)} total training pairs")
    
    # Create query groups (lists of 2 strings each)
    query_groups = create_query_groups(training_pairs)
    print(f"Created {len(query_groups)} query groups")
    
    # Run lambda sweep evaluation
    print("\n=== LAMBDA SWEEP EVALUATION ===")
    lambda_results_df = lambda_sweep_evaluation(query_groups)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_lambda_sweep_results(lambda_results_df)
    
    # Print summary
    print("\n=== LAMBDA SWEEP RESULTS ===")
    print("\nAccuracy by Lambda Value:")
    print(analysis['accuracy_by_lambda'])
    
    print("\nAccuracy by Lambda and Pair Type:")
    print(analysis['accuracy_by_lambda_type'])
    
    print("\nRank Distribution (when correct):")
    print(analysis['rank_distribution'])
    
    # Find best lambda
    best_lambda = analysis['accuracy_by_lambda']['accuracy'].idxmax()
    best_accuracy = analysis['accuracy_by_lambda']['accuracy'].max()
    print(f"\nBest Lambda: {best_lambda:.1f} with {best_accuracy:.2%} accuracy")
    
    # Save results
    save_lambda_sweep_results(analysis)
    
    return analysis

if __name__ == "__main__":
    # Run lambda sweep by default, or uncomment line below for original evaluation
    analysis = main_lambda_sweep()
    # query_results, accuracy_stats = main()