"""
Simplified solve_task for null hypothesis testing.
This version uses standard VAE KL divergence instead of sophisticated AWGN channel capacity calculations.
"""

import os
import time
import json
import gc
import traceback
import torch
import matplotlib.pyplot as plt

import preprocessing
import train_simplified  # Use our simplified training module
import arc_compressor
import solution_selection
import visualization


def save_kl_curves_plot(train_history_logger, task_name, save_dir="plots"):
    """
    Save the KL curves from training as a plot.
    (Same as original but for simplified training)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Find the starting point where total KL drops below 1000
    total_kl = train_history_logger.total_KL_curve
    start_idx = 0
    for i, kl_val in enumerate(total_kl):
        if kl_val < 1000:
            start_idx = i
            break
    
    # Top subplot: KL curves (filtered)
    for component_name, values in train_history_logger.KL_curves.items():
        filtered_values = values[start_idx:]
        if filtered_values:
            x_range = range(start_idx, start_idx + len(filtered_values))
            ax1.plot(x_range, filtered_values, label=f'KL {component_name}', alpha=0.7)
    
    # Plot total KL curve (filtered)
    filtered_total_kl = total_kl[start_idx:]
    if filtered_total_kl:
        x_range = range(start_idx, start_idx + len(filtered_total_kl))
        ax1.plot(x_range, filtered_total_kl, label='Total KL', linewidth=2, color='black')
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title(f'Simplified VAE KL Curves for Task: {task_name} (Values < 1000)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Loss and reconstruction error (also filtered from same start point)
    filtered_loss = train_history_logger.loss_curve[start_idx:]
    filtered_recon_error = train_history_logger.reconstruction_error_curve[start_idx:]
    
    if filtered_loss:
        x_range = range(start_idx, start_idx + len(filtered_loss))
        ax2.plot(x_range, filtered_loss, label='Total Loss', color='red', linewidth=2)
    
    if filtered_recon_error:
        x_range = range(start_idx, start_idx + len(filtered_recon_error))
        ax2.plot(x_range, filtered_recon_error, label='Reconstruction Error', color='blue', linewidth=2)
    
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Loss / Error')
    ax2.set_title(f'Simplified Training Progress for Task: {task_name} (From iteration {start_idx})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}_simplified_training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simplified training curves plot saved to: {save_path}")


def solve_task_simplified(
    task_name,
    split,
    time_limit,
    n_train_iterations,
    gpu_id,
    memory_dict,
    solutions_dict,
    error_queue,
    task_file=None,
):
    """
    Simplified version of solve_task using standard VAE KL divergence.
    
    This tests the null hypothesis that the sophisticated KL calculation 
    is not necessary for good performance.
    """
    try:
        # Get the task (same as original)
        task = None
        if not task_file:
            with open(f"dataset/arc-agi_{split}_challenges.json", "r") as f:
                problems = json.load(f)
            task = preprocessing.Task(task_name, problems[task_name], None)
            del problems
        else:
            with open(task_file, "r") as f:
                problem = json.load(f)
            task = preprocessing.Task(task_name, problem, None)

        # Set up the training (same as original)
        model = arc_compressor.ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        train_history_logger = solution_selection.Logger(task)
        train_history_logger.solution_most_frequent = tuple(
            ((0, 0), (0, 0)) for example_num in range(task.n_test)
        )
        train_history_logger.solution_second_most_frequent = tuple(
            ((0, 0), (0, 0)) for example_num in range(task.n_test)
        )

        # Create directory for iteration images
        os.makedirs(f"plots/{task_name}", exist_ok=True)

        # Training loop using simplified KL calculation
        for train_step in range(n_train_iterations):
            # Use simplified training step
            train_simplified.take_step_simplified(task, model, optimizer, train_step, train_history_logger)
            
            if train_step % 10 == 0:
                print(
                    "reconstruction error:",
                    train_history_logger.reconstruction_error_curve[-1],
                    "Simplified KL:",
                    train_history_logger.total_KL_curve[-1],
                )

                for key, value in train_history_logger.KL_curves.items():
                    print(f"  {key}: {value[-1]}")

            # Save attempts as images every 50th iteration
            if train_step % 50 == 0:
                fname = f"plots/{task_name}/{task_name}_simplified_at_{train_step}_steps.png"
                visualization.plot_solution(train_history_logger, fname=fname)

            if time.time() > time_limit:
                break

        # Save simplified training curves after training is complete
        save_kl_curves_plot(train_history_logger, task_name)

        # Get the solution (same as original)
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [
                list(row)
                for row in train_history_logger.solution_most_frequent[example_num]
            ]
            attempt_2 = [
                list(row)
                for row in train_history_logger.solution_second_most_frequent[
                    example_num
                ]
            ]
            example_list.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
        
        del task
        del model
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        # Store the result
        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list

    except Exception:
        error_queue.put(traceback.format_exc())
