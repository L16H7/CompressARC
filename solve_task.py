import pdb
import os
import time
import json
import gc
import traceback

import torch
import matplotlib.pyplot as plt

import preprocessing
import train
import arc_compressor
import solution_selection
import visualization

"""
A script that solves one puzzle, to be imported and used with parallel_train.py and multiprocessing.
"""


def save_kl_curves_plot(train_history_logger, task_name, save_dir="plots"):
    """
    Save the KL curves from training as a plot.

    Args:
        train_history_logger: Logger object containing KL_curves data
        task_name: Name of the task for file naming
        save_dir: Directory to save the plot (default: "plots")
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Find the starting point where total KL drops below 1000
    total_kl = train_history_logger.total_KL_curve
    start_idx = 0
    for i, kl_val in enumerate(total_kl):
        if kl_val < 1000:
            start_idx = i
            break

    allowed_kl_components = [
        "[0, 1, 0, 0, 0]",
        "[1, 1, 0, 0, 0]",
        "[0, 0, 1, 0, 0]",
        "[1, 0, 1, 0, 0]",
        "[1, 0, 0, 1, 0]",
        "[1, 0, 0, 0, 1]",
    ]
    # Top subplot: KL curves (filtered)
    for component_name, values in train_history_logger.KL_curves.items():
        filtered_values = values[start_idx:]
        if filtered_values:  # Only plot if there are values after filtering
            x_range = range(start_idx, start_idx + len(filtered_values))

            if component_name in allowed_kl_components:
                ax1.plot(x_range, filtered_values, label=f"KL {component_name}", alpha=0.7)

    # Plot total KL curve (filtered)
    filtered_total_kl = total_kl[start_idx:]
    if filtered_total_kl:
        x_range = range(start_idx, start_idx + len(filtered_total_kl))
        # ax1.plot(x_range, filtered_total_kl, label='Total KL', linewidth=2, color='black')

    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("KL Divergence")
    ax1.set_title(f"KL Curves for Task: {task_name} (Values < 1000)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Loss and reconstruction error (also filtered from same start point)
    filtered_loss = train_history_logger.loss_curve[start_idx:]
    filtered_recon_error = train_history_logger.reconstruction_error_curve[start_idx:]

    if filtered_loss:
        x_range = range(start_idx, start_idx + len(filtered_loss))
        ax2.plot(x_range, filtered_loss, label="Total Loss", color="red", linewidth=2)

    if filtered_recon_error:
        x_range = range(start_idx, start_idx + len(filtered_recon_error))
        ax2.plot(
            x_range,
            filtered_recon_error,
            label="Reconstruction Error",
            color="blue",
            linewidth=2,
        )

    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Loss / Error")
    ax2.set_title(
        f"Training Progress for Task: {task_name} (From iteration {start_idx})"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}_training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves plot saved to: {save_path}")


def solve_task(
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
    Solves a puzzle.
    Args:
        task_name (str): The name of the puzzle to solve.
        split (str): 'training', 'evaluation', or 'test'
        time_limit (float): An end time that will cause training to exit early if reached.
        n_train_iterations (int): The number of iterations to train for.
        gpu_id (int): The GPU number to run the solver on.
        memory_dict (multiprocessing.Dict[str, int]): An inter-process shared dict that we
            can store the amount of memory taken by this job in.
        solutions_dict (multiprocessing.Dict[str, list[Dict[str, list[list[int]]]]]): An
            inter-process shared dict that we can store the solution in.
        error_queue (multiprocessing.Queue[Exception]): An inter-process shared queue to
            put errors in when an exception occurs.
    """

    try:  # Error catching block that puts errors on the error_queue
        # torch.set_default_device('cuda')
        # torch.cuda.set_device(gpu_id)
        # torch.cuda.reset_peak_memory_stats()  # Measure the memory used.

        # Get the task
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

        # Set up the training
        model = arc_compressor.ARCCompressor(task)
        # import pdb; pdb.set_trace()  # Debugging breakpoint
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

        # Training loop
        for train_step in range(n_train_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger)
            if train_step % 10 == 0:
                print(
                    "reconstruction error:",
                    train_history_logger.reconstruction_error_curve[-1],
                    "KL",
                    train_history_logger.total_KL_curve[-1],
                )
                # pdb.set_trace()

                # for key, value in train_history_logger.KL_curves.items():
                # print(f"  {key}: {value[-1]}")

            # Save attempts as images every 10th iteration
            if train_step % 50 == 0:
                fname = f"plots/{task_name}/{task_name}_at_{train_step}_steps.png"
                visualization.plot_solution(train_history_logger, fname=fname)

            # pdb.set_trace()
            if time.time() > time_limit:
                break

        # Save training curves (KL curves, loss, reconstruction error) after training is complete
        save_kl_curves_plot(train_history_logger, task_name)

        # Get the solution
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

    except Exception:  # If error, write to the error queue
        error_queue.put(traceback.format_exc())
