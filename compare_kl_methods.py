#!/usr/bin/env python3
"""
Comparison script to test both original and simplified KL calculations.
This helps determine if the sophisticated KL calculation provides benefits.
"""

import time
import multiprocessing
import torch
import matplotlib.pyplot as plt
import numpy as np
from solve_task import solve_task
from solve_task_simplified import solve_task_simplified


def run_comparison(task_name="007bbfb7", n_iterations=200, task_file=None):
    """
    Run both original and simplified versions for comparison.
    """
    
    print("=== COMPARATIVE NULL HYPOTHESIS TEST ===")
    print(f"Task: {task_name}")
    print(f"Iterations: {n_iterations}")
    print("Comparing:")
    print("1. Original: Sophisticated AWGN channel capacity KL calculation")
    print("2. Simplified: Standard VAE KL = 0.5 * (μ² + σ² - log(σ²) - 1)")
    print()
    
    # Common parameters
    split = "training"
    time_limit = time.time() + 30000  # Large time limit
    gpu_id = 0
    
    results = {}
    
    # Test original version
    print("=== RUNNING ORIGINAL VERSION ===")
    try:
        manager = multiprocessing.Manager()
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        
        start_time = time.time()
        solve_task(
            task_name=task_name,
            split=split,
            time_limit=time_limit,
            n_train_iterations=n_iterations,
            gpu_id=gpu_id,
            memory_dict=memory_dict,
            solutions_dict=solutions_dict,
            error_queue=error_queue,
            task_file=task_file
        )
        end_time = time.time()
        
        results['original'] = {
            'time': end_time - start_time,
            'memory': memory_dict.get(task_name, 0),
            'solutions': solutions_dict.get(task_name, []),
            'errors': []
        }
        
        while not error_queue.empty():
            results['original']['errors'].append(error_queue.get())
            
        print(f"Original completed in {results['original']['time']:.2f} seconds")
        
    except Exception as e:
        print(f"Original version failed: {e}")
        results['original'] = {'error': str(e)}
    
    # Test simplified version
    print("\n=== RUNNING SIMPLIFIED VERSION ===")
    try:
        manager = multiprocessing.Manager()
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        
        start_time = time.time()
        solve_task_simplified(
            task_name=task_name,
            split=split,
            time_limit=time_limit,
            n_train_iterations=n_iterations,
            gpu_id=gpu_id,
            memory_dict=memory_dict,
            solutions_dict=solutions_dict,
            error_queue=error_queue,
            task_file=task_file
        )
        end_time = time.time()
        
        results['simplified'] = {
            'time': end_time - start_time,
            'memory': memory_dict.get(task_name, 0),
            'solutions': solutions_dict.get(task_name, []),
            'errors': []
        }
        
        while not error_queue.empty():
            results['simplified']['errors'].append(error_queue.get())
            
        print(f"Simplified completed in {results['simplified']['time']:.2f} seconds")
        
    except Exception as e:
        print(f"Simplified version failed: {e}")
        results['simplified'] = {'error': str(e)}
    
    # Print comparison results
    print("\n=== COMPARISON RESULTS ===")
    
    if 'error' not in results.get('original', {}) and 'error' not in results.get('simplified', {}):
        print(f"Training Time:")
        print(f"  Original:   {results['original']['time']:.2f} seconds")
        print(f"  Simplified: {results['simplified']['time']:.2f} seconds")
        print(f"  Difference: {results['simplified']['time'] - results['original']['time']:.2f} seconds")
        print()
        
        print(f"Memory Usage:")
        print(f"  Original:   {results['original']['memory']:,} bytes")
        print(f"  Simplified: {results['simplified']['memory']:,} bytes")
        print()
        
        print(f"Solutions Generated:")
        print(f"  Original:   {len(results['original']['solutions'])}")
        print(f"  Simplified: {len(results['simplified']['solutions'])}")
        print()
        
        print("Error Status:")
        print(f"  Original errors:   {len(results['original']['errors'])}")
        print(f"  Simplified errors: {len(results['simplified']['errors'])}")
        print()
        
        print("Generated Files for Analysis:")
        print(f"  Original curves:   plots/{task_name}_training_curves.png")
        print(f"  Simplified curves: plots/{task_name}_simplified_training_curves.png")
        print(f"  Original solutions: plots/{task_name}/{task_name}_at_*_steps.png")
        print(f"  Simplified solutions: plots/{task_name}/{task_name}_simplified_at_*_steps.png")
        print()
        
        print("CONCLUSION:")
        if len(results['original']['errors']) == 0 and len(results['simplified']['errors']) == 0:
            print("✓ Both versions completed successfully")
            print("✓ Compare the generated plots to see differences in:")
            print("  - Convergence speed")
            print("  - Final KL values")
            print("  - Solution quality")
            print("  - Training stability")
        else:
            print("⚠ One or both versions had errors - check error details")
            
    else:
        print("❌ One or both versions failed to complete")
        if 'error' in results.get('original', {}):
            print(f"Original error: {results['original']['error']}")
        if 'error' in results.get('simplified', {}):
            print(f"Simplified error: {results['simplified']['error']}")
    
    return results


if __name__ == "__main__":
    results = run_comparison(
        task_name="007bbfb7",
        n_iterations=200,
        task_file="/Users/light/projects/CompressARC/dataset/simple_connect.json"
    )
