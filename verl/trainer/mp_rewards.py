from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Optional
import time
import signal
import os
from tqdm import tqdm
import traceback
import gc
import psutil
from functools import wraps
from verl import DataProto
from verl.utils.reward_score import code_prime, math_prime

def timeout_wrapper(func, timeout):
    """Wrapper to add timeout to any function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None
    return wrapper

class ProgressTracker:
    """Simple progress tracker using tqdm"""
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.pbar = tqdm(total=total, desc=desc)
        
    def update(self, n: int = 1):
        self.pbar.update(n)
        
    def close(self):
        self.pbar.close()

def process_chunk(chunk: List[Dict], start_time: float, timeout: int) -> List[Dict]:
    """Process a chunk of items with timeout based on start_time"""
    results = []
    for item in chunk:
        if time.time() - start_time > timeout:
            break
        result = process_item(item, start_time, timeout)
        item.clear()  # Clear processed item
        results.append(result)
    return results

def process_item(item_dict: Dict, start_time: float, timeout: int) -> Dict:
    """Process a single item with timeout check"""
    try:
        # Early validation
        if not all(k in item_dict for k in ['sequence_str', 'ground_truth', 'ability']):
            raise ValueError("Missing required fields in item_dict")
            
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError("Global timeout reached")

        # Get metadata and sequence
        sequences_str = item_dict['sequence_str']
        ground_truth = item_dict['ground_truth']
        data_ability = item_dict['ability']
        
        # Select scoring function
        if data_ability == 'math':
            score_fn = math_prime.compute_score
        elif data_ability == 'code':
            score_fn = code_prime.compute_score
        else:
            raise ValueError(f"Unknown ability: {data_ability}")
            
        # Compute score
        score = score_fn(sequences_str, ground_truth)
        
        # Clear large strings
        sequences_str = None
        ground_truth = None
        gc.collect()
        
        return {
            'batch_idx': item_dict['batch_idx'],
            'response_length': item_dict['response_length'],
            'score': float(score),
            'success': True,
            'error': None
        }
    except TimeoutError:
        return {
            'batch_idx': item_dict['batch_idx'],
            'success': False,
            'error': 'Global timeout reached'
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            'batch_idx': item_dict['batch_idx'],
            'success': False,
            'error': f"{str(e)}\n{error_traceback}"
        }

def kill_process_tree(pid):
    """Kill a process and its children"""
    try:
        # Check if PID exists and get process
        proc = psutil.Process(pid)
        
        # Get children first
        children = proc.children(recursive=True)
        
        # Kill children
        for child in children:
            try:
                os.kill(child.pid, signal.SIGKILL)
            except:
                pass
                
        # Kill parent
        os.kill(pid, signal.SIGKILL)
    except:
        pass

class RewardManager:
    def __init__(self, 
                 tokenizer, 
                 num_examine: int = 5, 
                 debug_level: int = 1, 
                 min_chunk_size: int = 5, 
                 timeout: int = 360):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.debug_level = debug_level
        self.min_chunk_size = min_chunk_size
        self.timeout = timeout
        self.num_processes = max(1, mp.cpu_count() - 2)
        
    def prepare_item(self, idx: int, data_item) -> Optional[Dict]:
        """Prepare a single item's data in the main process"""
        try:
            # Extract data (in main process)
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Concatenate and decode in main process
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # Clear tensors to save memory
            del sequences
            
            return {
                'batch_idx': idx,
                'sequence_str': sequences_str,
                'response_length': len(valid_response_ids),
                'ground_truth': data_item.non_tensor_batch['reward_model']['ground_truth'],
                'ability': data_item.non_tensor_batch['ability']
            }
        except Exception as e:
            print(f"Failed to prepare item {idx}: {str(e)}")
            return None

    def cleanup_processes(self, executor, futures):
        """Cleanup processes with timeout protection"""
        # Cancel pending futures
        for future in futures:
            future.cancel()
            
        # Shutdown executor with timeout
        shutdown_with_timeout = timeout_wrapper(executor.shutdown, timeout=5)
        shutdown_with_timeout(wait=False)
        
        # Kill any remaining processes
        for proc in mp.active_children():
            kill_process_tree(proc.pid)
            
        gc.collect()

    def __call__(self, data: DataProto) -> torch.Tensor:
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
            
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        try:
            if self.debug_level >= 1:
                print(f"\nProcessing {len(data)} items using {self.num_processes} processes")
                print(f"Global timeout set to {self.timeout}s")
            
            # Prepare items
            items = []
            prep_progress = ProgressTracker(len(data), "Preparing items")
            for i in range(len(data)):
                item_dict = self.prepare_item(i, data[i])
                if item_dict is not None:
                    items.append(item_dict)
                prep_progress.update()
            prep_progress.close()
            
            if not items:
                raise ValueError("No valid items to process")
            
            # Calculate chunk size
            chunk_size = max(
                self.min_chunk_size,
                min(50, len(items) // (self.num_processes * 2))
            )
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            if self.debug_level >= 1:
                print(f"Processing in {len(chunks)} chunks of size ~{chunk_size}")
                
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all jobs
                futures = [
                    executor.submit(process_chunk, chunk, start_time, self.timeout)
                    for chunk in chunks
                ]
                
                # Process futures
                remaining_futures = futures
                while remaining_futures:
                    # Check timeout first
                    if time.time() - start_time > self.timeout:
                        print(f"\nGlobal timeout of {self.timeout}s reached")
                        self.cleanup_processes(executor, remaining_futures)
                        break
                        
                    # Wait with shorter timeout
                    done, remaining_futures = wait(
                        remaining_futures, 
                        timeout=min(1.0, max(0.1, self.timeout - (time.time() - start_time)))
                    )
                    
                    # Process completed futures
                    for future in done:
                        try:
                            chunk_results = future.result(timeout=0.1)
                            for result in chunk_results:
                                if result['success']:
                                    batch_idx = result['batch_idx']
                                    response_idx = result['response_length'] - 1
                                    reward_tensor[batch_idx, response_idx] = result['score']
                        except Exception as e:
                            print(f"Error processing chunk result: {str(e)}")
                            continue
                            
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Ensure cleanup
            gc.collect()
            
        return reward_tensor