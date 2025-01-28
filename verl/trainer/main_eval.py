# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""

import hydra
from concurrent.futures import ThreadPoolExecutor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import code_prime, math_prime
import pandas as pd
import numpy as np


def select_reward_fn(data_ability):
    if data_ability == 'math':
        return math_prime.compute_score
    elif data_ability == 'code':
        return code_prime.compute_score
    else:
        raise NotImplementedError


def process_single_item(args):
    """Process a single evaluation item in a thread."""
    response_lst, data_ability, prompt, reward_data = args
    
    # Select reward function
    reward_fn = select_reward_fn(data_ability)
    ground_truth = reward_data['ground_truth']
    
    # Compute scores for all responses
    score_lst = []
    for r in response_lst:
        score = reward_fn(r, ground_truth)
        score_lst.append(score)
    
    return np.max(score_lst)


class ThreadedEvaluator:
    """Handles threaded evaluation of responses."""
    
    def __init__(self, num_threads=512):
        self.num_threads = num_threads
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
    
    def __del__(self):
        """Cleanup thread pool."""
        self._executor.shutdown(wait=False)
    
    def evaluate(self, dataset, config):
        """
        Evaluate the dataset using multiple threads.
        
        Args:
            dataset: Pandas DataFrame containing the evaluation data
            config: Hydra config object containing data keys
        
        Returns:
            float: pass@5 score
        """
        prompts = dataset[config.data.prompt_key]
        responses = dataset[config.data.response_key]
        reward_model_data = dataset[config.data.reward_model_key]
        data_abilities = dataset["ability"]
        
        total = len(dataset)
        
        try:
            # Submit all tasks to thread pool
            futures = [
                self._executor.submit(
                    process_single_item,
                    (responses[i], data_abilities[i], prompts[i], reward_model_data[i])
                )
                for i in range(total)
            ]
            
            # Gather results as they complete
            max_scores = [future.result() for future in futures]
            
        except Exception as e:
            print(f"Parallel processing failed with error: {str(e)}")
            print("Falling back to sequential processing")
            
            # Sequential fallback
            max_scores = []
            for i in range(total):
                score = process_single_item(
                    (responses[i], data_abilities[i], prompts[i], reward_model_data[i])
                )
                max_scores.append(score)
        
        # Calculate passes (score == 1)
        passes = sum(1 for score in max_scores if score == 1)
        
        return passes / total


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    # Load dataset
    local_path = copy_local_path_from_hdfs(config.data.path)
    dataset = pd.read_parquet(local_path)
    
    # Create evaluator and run evaluation
    evaluator = ThreadedEvaluator(num_threads=512)
    pass_rate = evaluator.evaluate(dataset, config)
    
    print(f'pass@5: {pass_rate}')


if __name__ == '__main__':
    main()