#!/usr/bin/env python3
"""
Test script for the simplified solve_task function (null hypothesis testing).
This tests whether the sophisticated KL calculation is necessary for good performance.
"""

import time
import multiprocessing
import torch
from solve_task_simplified import solve_task_simplified


def main():
    """
    Main function to test the simplified solve_task for null hypothesis testing.
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This may cause issues since solve_task expects GPU.")
        gpu_id = 0  # Will still set to 0, but expect potential errors
    else:
        gpu_id = 0  # Use first GPU
        print(f"Using GPU {gpu_id}")
    
    # Set up parameters
    task_name = "007bbfb7"  # First task from the training set
    split = "training"  # Use training split
    time_limit = time.time() + 30000  # Long time limit for complete test
    n_train_iterations = 200  # Small number for testing
    
    # Create multiprocessing objects
    manager = multiprocessing.Manager()
    memory_dict = manager.dict()
    solutions_dict = manager.dict()
    error_queue = manager.Queue()
    
    print("=== NULL HYPOTHESIS TEST ===")
    print("Testing simplified VAE KL divergence vs sophisticated AWGN channel capacity calculation")
    print()
    print("Starting simplified solve_task with:")
    print(f"  task_name: {task_name}")
    print(f"  split: {split}")
    print(f"  time_limit: {time_limit}")
    print(f"  n_train_iterations: {n_train_iterations}")
    print(f"  gpu_id: {gpu_id}")
    print()
    print("This version uses standard VAE KL = 0.5 * (μ² + σ² - log(σ²) - 1)")
    print("Instead of sophisticated AWGN channel capacity formulas")
    print()
    
    # Call simplified solve_task
    try:
        start_time = time.time()
        
        solve_task_simplified(
            task_name=task_name,
            split=split,
            time_limit=time_limit,
            n_train_iterations=n_train_iterations,
            gpu_id=gpu_id,
            memory_dict=memory_dict,
            solutions_dict=solutions_dict,
            error_queue=error_queue,
            task_file="/Users/light/projects/CompressARC/dataset/simple_connect.json"
        )
        
        end_time = time.time()
        
        print("\n=== SIMPLIFIED SOLVE_TASK COMPLETED ===")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        print(f"Memory used: {memory_dict.get(task_name, 'N/A')} bytes")
        print(f"Solutions generated: {len(solutions_dict.get(task_name, []))}")
        
        # Check for any errors
        if not error_queue.empty():
            print("\nErrors found:")
            while not error_queue.empty():
                error = error_queue.get()
                print(error)
        else:
            print("\nNo errors found!")
            print("\nNULL HYPOTHESIS TEST RESULTS:")
            print("- Simplified VAE KL divergence calculation worked")
            print("- Training completed without sophisticated AWGN formulas")
            print("- Check the generated plots to compare:")
            print(f"  - plots/{task_name}_simplified_training_curves.png")
            print(f"  - plots/{task_name}/{task_name}_simplified_at_*_steps.png")
            print()
            print("Next steps for analysis:")
            print("1. Compare simplified vs original training curves")
            print("2. Compare final solution quality")
            print("3. Compare convergence speed and stability")
        
    except Exception as e:
        print(f"Error calling simplified solve_task: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
