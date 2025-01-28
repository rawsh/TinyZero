import os
import datasets
import argparse
import json
from transformers import AutoTokenizer
from itertools import islice

def load_tokenizer(model_name):
    """Load tokenizer for the specified model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def batch_iterator(iterable, batch_size):
    """Yield batches from an iterable"""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def filter_batch_tokens(batch, tokenizer, max_tokens):
    """Filter a batch of examples based on token length"""
    if not tokenizer:
        return [True] * len(batch)
    
    # Batch tokenize all prompts
    prompts = [str(example["prompt"]) for example in batch]
    encodings = tokenizer(prompts, padding=False, truncation=False)
    
    # Check token lengths
    return [len(tokens) <= max_tokens for tokens in encodings['input_ids']]

def filter_content(example):
    """Filter examples based on content criteria"""
    # Code ability filter
    if example['ability'] == 'code':
        try:
            parsed_gt = json.loads(example["reward_model"]["ground_truth"])
            if 'inputs' in parsed_gt and 'outputs' in parsed_gt:
                if len(parsed_gt['inputs']) > 0 and len(parsed_gt['outputs']) > 0:
                    return True
                else:
                    print(f"Warning: example is empty")
                    return False
            else:
                print(f"Warning: example has no inputs or outputs")
                return False
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON in ground truth")
            return False

    # Math ability filter
    elif example['ability'] == 'math':
        gt = example["reward_model"]["ground_truth"]

        # Basic content filters
        if gt == "":
            return False
        
        bad = ["mm/s", "cm/s", "m/s", "km/s", "km/h", "mph"]
        if any(b in gt for b in bad):
            return False
        
        if "\\boxed" in gt:
            return False

        return True
    
    return False

def filter_examples(dataset, tokenizer=None, max_tokens=1000, batch_size=1024):
    """Filter dataset based on conditions including batched token length checks"""
    # First filter based on content (non-token criteria)
    content_filtered = dataset.filter(filter_content)
    
    if not tokenizer:
        return content_filtered
    
    # Then filter based on token length using batched processing
    keep_indices = []
    all_indices = range(len(content_filtered))
    
    for batch_idx in batch_iterator(all_indices, batch_size):
        batch_examples = [content_filtered[i] for i in batch_idx]
        batch_results = filter_batch_tokens(batch_examples, tokenizer, max_tokens)
        
        # Keep track of indices to keep
        keep_indices.extend([idx for idx, keep in zip(batch_idx, batch_results) if keep])

    # debug log
    if len(keep_indices) != len(all_indices):
        print("filtered by length: ", len(all_indices) - len(keep_indices))
    
    # Select only the examples we want to keep
    return content_filtered.select(keep_indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/filtered')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer is None:
        print("Warning: Proceeding without tokenizer-based filtering")

    # Load Eurus dataset
    data_source = 'rawsh/Eurus-2-RL-Data-ProblemsOnly'
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    print(f"Original sizes - train: {len(dataset['train'])}, val: {len(dataset['validation'])}")

    # Filter datasets
    train_dataset = filter_examples(dataset['train'], tokenizer, args.max_tokens, args.batch_size)
    val_dataset = filter_examples(dataset['validation'], tokenizer, args.max_tokens, args.batch_size)

    print(f"Filtered sizes - train: {len(train_dataset)}, val: {len(val_dataset)}")

    # Save to parquet
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS utilities not available. Skipping HDFS copy.")

if __name__ == '__main__':
    main()