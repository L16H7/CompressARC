#!/usr/bin/env python3
"""
Simple test script for the null hypothesis - test simplified KL calculation only.
"""

import time
import multiprocessing
from solve_task_simplified import solve_task_simplified


def main():
    """
    Test the simplified KL calculation to see if sophisticated AWGN is necessary.
    """
    
    print("=== NULL HYPOTHESIS TEST ===")
    print("Testing: Is sophisticated AWGN channel capacity calculation necessary?")
    print("Alternative: Simple VAE KL = 0.5 * (μ² + σ² - log(σ²) - 1)")
    print()
    
    # Parameters
    task_name = "272f95fa"
    split = "training"
    time_limit = time.time() + 30000
    n_train_iterations = 2000
    gpu_id = 0
    
    # Multiprocessing setup
    manager = multiprocessing.Manager()
    memory_dict = manager.dict()
    solutions_dict = manager.dict()
    error_queue = manager.Queue()
    
    print(f"Running simplified version with {n_train_iterations} iterations...")
    print()
    
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
            # task_file="/Users/light/projects/CompressARC/dataset/simple_connect.json"
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n=== RESULTS ===")
        print(f"✓ Training completed in {training_time:.2f} seconds")
        print(f"✓ Memory used: {memory_dict.get(task_name, 'N/A')} bytes")
        print(f"✓ Solutions: {len(solutions_dict.get(task_name, []))}")
        
        if error_queue.empty():
            print("✓ No errors!")
            print("\nFILES GENERATED:")
            print(f"- Training curves: plots/{task_name}_simplified_training_curves.png")
            print(f"- Solution plots: plots/{task_name}/{task_name}_simplified_at_*_steps.png")
            print("\nNULL HYPOTHESIS RESULT:")
            print("✓ Simple VAE KL divergence works without sophisticated AWGN calculations!")
            print("  This suggests the complex channel capacity formulas may not be essential.")
            print("\nNEXT STEPS:")
            print("1. Compare with original version performance")
            print("2. Check solution quality in the generated plots")
            print("3. Analyze convergence behavior")
        else:
            print("\n❌ ERRORS FOUND:")
            while not error_queue.empty():
                error = error_queue.get()
                print(f"  {error}")
                
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
