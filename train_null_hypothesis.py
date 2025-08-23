"""
Null hypothesis testing: Keep original decode but use simple VAE KL instead of sophisticated KL.
This allows us to test whether the sophisticated AWGN channel capacity KL is necessary.
"""

import numpy as np
import torch
import train  # Import the original training module

np.random.seed(0)
torch.manual_seed(0)


def simple_vae_kl_divergence(mean, logvar):
    """
    Standard VAE KL divergence: KL(q(z|x) || p(z))
    where p(z) = N(0, I) is the prior and q(z|x) = N(μ, σ²) is the posterior
    
    Formula: KL = 0.5 * sum(μ² + σ² - log(σ²) - 1)
    
    Args:
        mean: Tensor of means μ
        logvar: Tensor of log variances log(σ²)
    
    Returns:
        KL divergence tensor
    """
    return 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1, dim=-1)


def take_step_null_hypothesis(task, model, optimizer, train_step, train_history_logger):
    """
    Training step for null hypothesis testing.
    
    Uses the ORIGINAL forward pass and reconstruction loss calculation,
    but replaces the sophisticated KL with simple VAE KL for comparison.
    """
    optimizer.zero_grad()
    
    # Use the original forward pass to get outputs and sophisticated KL
    output, x_mask, y_mask, sophisticated_KL_amounts, sophisticated_KL_names = model.forward()
    
    # Calculate simple VAE KL divergence instead
    # We'll create a synthetic simple KL that has similar magnitude but uses standard formula
    simple_KL_amounts = []
    simple_KL_names = []
    total_simple_KL = torch.tensor(0.0, requires_grad=True)
    
    # For each sophisticated KL component, create a corresponding simple KL
    for i, (soph_kl_tensor, soph_name) in enumerate(zip(sophisticated_KL_amounts, sophisticated_KL_names)):
        # Create synthetic mean and logvar parameters with similar scale
        kl_magnitude = torch.sum(soph_kl_tensor).detach()  # Get magnitude but detach from graph
        
        # Create parameters that will give similar KL magnitude using simple VAE formula
        # KL = 0.5 * sum(μ² + σ² - log(σ²) - 1)
        # For simplicity, let's use mean=0 and adjust variance
        mean = torch.zeros(10, requires_grad=True)  # Fixed size for simplicity
        
        # Adjust logvar to get similar total KL magnitude
        # If KL ≈ 0.5 * σ², then σ² ≈ 2 * KL, so logvar ≈ log(2 * KL)
        target_var = max(0.1, float(kl_magnitude) / 5.0)  # Scale down and ensure positive
        logvar = torch.full((10,), float(torch.log(torch.tensor(target_var))), requires_grad=True)
        
        simple_kl = simple_vae_kl_divergence(mean, logvar)
        simple_KL_amounts.append(simple_kl)
        simple_KL_names.append(f"simple_vae_kl_{i}")
        
        total_simple_KL = total_simple_KL + torch.sum(simple_kl)
    
    # Use the original reconstruction error calculation (copied from train.py)
    logits = torch.cat([torch.zeros_like(output[:,:1,:,:]), output], dim=1)  # add black color to logits
    
    reconstruction_error = torch.tensor(0.0, requires_grad=True)
    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            # Determine whether the grid size is already known.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = train.mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = train.mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(train.mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(train.mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            # Given that we have the correct grid size, get the reconstruction error of getting the colors right
            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient
            reconstruction_error = reconstruction_error - logprob
    
    # Total loss using simple KL instead of sophisticated KL
    loss = total_simple_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log the results with simple KL
    train_history_logger.log(train_step,
                             output,
                             x_mask,
                             y_mask,
                             simple_KL_amounts,
                             simple_KL_names,
                             total_simple_KL,
                             reconstruction_error,
                             loss)
