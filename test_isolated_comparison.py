#!/usr/bin/env python3
"""
Fixed comparison script with proper random seed management.
"""

import time
import multiprocessing
import subprocess
import sys


def run_test_isolated(script_name, description):
    """
    Run a test script in a separate Python process to avoid state contamination.
    """
    print(f"\n=== {description} ===")
    print(f"Running: {script_name}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Runtime: {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✓ SUCCESS")
            # Print last few lines of output for summary
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 5:
                print("Output summary:")
                for line in output_lines[-5:]:
                    print(f"  {line}")
            else:
                print("Full output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            
        return {
            'success': result.returncode == 0,
            'runtime': end_time - start_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (>5 minutes)")
        return {'success': False, 'runtime': 300, 'error': 'timeout'}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {'success': False, 'runtime': 0, 'error': str(e)}


def main():
    """
    Run both tests in isolation to ensure fair comparison.
    """
    
    print("=== ISOLATED NULL HYPOTHESIS COMPARISON ===")
    print("Testing both original and simplified KL calculations")
    print("Each test runs in a separate Python process to avoid state contamination")
    print()
    
    # Test 1: Original version alone (using test script that calls original)
    print("Creating temporary test script for original version...")
    
    # Create a temporary script for testing original version
    original_test_script = """
import time
import multiprocessing
from solve_task import solve_task

def main():
    task_name = "007bbfb7"
    split = "training"
    time_limit = time.time() + 30000
    n_train_iterations = 200
    gpu_id = 0
    
    manager = multiprocessing.Manager()
    memory_dict = manager.dict()
    solutions_dict = manager.dict()
    error_queue = manager.Queue()
    
    print("=== ORIGINAL VERSION TEST ===")
    print(f"Running original solve_task with {n_train_iterations} iterations...")
    
    try:
        start_time = time.time()
        solve_task(
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
        
        print(f"\\n=== ORIGINAL RESULTS ===")
        print(f"✓ Training completed in {end_time - start_time:.2f} seconds")
        print(f"✓ Memory used: {memory_dict.get(task_name, 'N/A')} bytes")
        print(f"✓ Solutions: {len(solutions_dict.get(task_name, []))}")
        
        if error_queue.empty():
            print("✓ No errors!")
        else:
            print("❌ ERRORS:")
            while not error_queue.empty():
                print(f"  {error_queue.get()}")
                
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
    
    with open("temp_test_original.py", "w") as f:
        f.write(original_test_script)
    
    # Run tests
    results = {}
    
    # Test original version
    results['original'] = run_test_isolated("temp_test_original.py", "ORIGINAL VERSION TEST")
    
    # Test simplified version  
    results['simplified'] = run_test_isolated("test_null_hypothesis.py", "SIMPLIFIED VERSION TEST")
    
    # Cleanup
    import os
    try:
        os.remove("temp_test_original.py")
    except:
        pass
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    for version, result in results.items():
        print(f"\n{version.upper()} VERSION:")
        if result['success']:
            print(f"  ✓ SUCCESS - Runtime: {result['runtime']:.2f}s")
        else:
            print(f"  ❌ FAILED - Runtime: {result['runtime']:.2f}s")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    print("\nFILES TO COMPARE:")
    print("- Original training curves: plots/007bbfb7_training_curves.png")
    print("- Simplified training curves: plots/007bbfb7_simplified_training_curves.png") 
    print("- Original solutions: plots/007bbfb7/007bbfb7_at_*_steps.png")
    print("- Simplified solutions: plots/007bbfb7/007bbfb7_simplified_at_*_steps.png")
    
    print("\nNULL HYPOTHESIS CONCLUSION:")
    if results['original']['success'] and results['simplified']['success']:
        print("✓ Both versions completed - compare the output files to assess performance")
        print("✓ This proves simplified KL divergence CAN work")
        print("? Check solution quality to determine if sophisticated KL provides benefits")
    elif results['simplified']['success'] and not results['original']['success']:
        print("🤔 Simplified version succeeded while original failed - interesting!")
        print("✓ This supports the null hypothesis that sophisticated KL may be unnecessary")
    elif results['original']['success'] and not results['simplified']['success']:
        print("❌ Original succeeded but simplified failed")
        print("❌ This suggests sophisticated KL calculation may be necessary")
    else:
        print("❌ Both versions failed - need to debug the setup")


if __name__ == "__main__":
    main()
