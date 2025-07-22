#!/usr/bin/env python3
"""
Test script to trace the solve_task function with appropriate parameters.
"""

import time
import multiprocessing
import torch
from solve_task import solve_task


def main():
    """
    Main function to call solve_task for tracing/debugging.
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This may cause issues since solve_task expects GPU.")
        gpu_id = 0  # Will still set to 0, but expect potential errors
    else:
        gpu_id = 0  # Use first GPU
        print(f"Using GPU {gpu_id}")
    
    # Set up parameters
    # task_name = "007bbfb7"  # First task from the training set
    task_name = "2281f1f4"  # First task from the training set
    split = "training"  # Use training split since we know the task exists there
    time_limit = time.time() + 30000  # 5 minutes from now
    n_train_iterations = 1_000_000 # Small number for testing
    
    # Create multiprocessing objects
    manager = multiprocessing.Manager()
    memory_dict = manager.dict()
    solutions_dict = manager.dict()
    error_queue = manager.Queue()
    
    print("Starting solve_task with:")
    print(f"  task_name: {task_name}")
    print(f"  split: {split}")
    print(f"  time_limit: {time_limit}")
    print(f"  n_train_iterations: {n_train_iterations}")
    print(f"  gpu_id: {gpu_id}")
    print()
    
    # Call solve_task
    try:
        solve_task(
            task_name=task_name,
            split=split,
            time_limit=time_limit,
            n_train_iterations=n_train_iterations,
            gpu_id=gpu_id,
            memory_dict=memory_dict,
            solutions_dict=solutions_dict,
            error_queue=error_queue
        )
        
        print("solve_task completed successfully!")
        print(f"Memory used: {memory_dict.get(task_name, 'N/A')} bytes")
        print(f"Solutions generated: {len(solutions_dict.get(task_name, []))}")
        
        # Check for any errors
        if not error_queue.empty():
            print("\nErrors found:")
            while not error_queue.empty():
                error = error_queue.get()
                print(error)
        
    except Exception as e:
        print(f"Error calling solve_task: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
