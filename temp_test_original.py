
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
        
        print(f"\n=== ORIGINAL RESULTS ===")
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
