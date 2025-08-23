"""
PROPERLY simplified training module for null hypothesis testing.
This keeps the EXACT same decoding but ONLY replaces KL calculation with simple VAE KL.
"""

import numpy as np
import torch
import layers
import multitensor_systems

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


def simple_channel_layer_kl_only(target_capacity, posterior):
    """
    Use the ORIGINAL channel layer for decoding but replace ONLY the KL calculation.
    This ensures identical decoding but different KL.
    """
    mean, local_capacity_adjustment = posterior

    # Use the ORIGINAL sophisticated decoding logic from layers.channel_layer
    all_but_last_dim = tuple(range(len(mean.shape)-1))
    dimensionality = 1
    for axis_length in mean.shape:
        dimensionality *= axis_length
    min_capacity = 0.5
    init_capacity = 10000
    min_capacity = torch.tensor(min_capacity)
    init_capacity = torch.tensor(init_capacity)

    target_capacity = 10*target_capacity

    # ORIGINAL output scaling computation
    desired_global_capacity = torch.exp(target_capacity)*init_capacity + min_capacity
    output_scaling = 1-torch.exp(-desired_global_capacity / dimensionality * 2)

    # ORIGINAL local adjustments
    local_capacity_adjustment = (target_capacity + 
                                 local_capacity_adjustment - 
                                 torch.mean(local_capacity_adjustment, dim=all_but_last_dim))
    desired_local_capacity = torch.exp(local_capacity_adjustment)*init_capacity + min_capacity

    # ORIGINAL signal/noise computation
    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std**2
    stable_sqrt1memx = lambda x: torch.where(x>20, 1, torch.sqrt(1-torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    signal_var = 1-noise_var

    # ORIGINAL normalization
    normalized_mean = mean - torch.mean(mean, dim=all_but_last_dim)
    normalized_mean = normalized_mean / torch.sqrt(torch.mean(normalized_mean**2+1e-8, dim=all_but_last_dim))

    # ORIGINAL sampling (KEEP THIS THE SAME!)
    z = signal_std*normalized_mean + noise_std*torch.randn(normalized_mean.shape)
    z = output_scaling*z

    # HERE IS THE ONLY CHANGE: Replace sophisticated KL with simple VAE KL
    # Convert the sophisticated parameters to simple VAE format
    logvar = torch.log(noise_var)  # Convert noise variance to log variance
    simple_kl = simple_vae_kl_divergence(normalized_mean, logvar)
    
    return z, simple_kl


def simple_decode_latents_kl_only(target_capacities, decode_weights, multiposteriors):
    """
    Use ORIGINAL decode structure but replace ONLY the KL calculation.
    """
    KL_amounts = []
    KL_names = []

    @multitensor_systems.multify
    def decode_latents_simple_kl(dims, target_capacity, decode_weight, posterior):
        # Use our modified channel layer that keeps decoding but changes KL
        z, simple_KL = simple_channel_layer_kl_only(target_capacity, posterior)
        # Use ORIGINAL affine layer
        x = layers.affine(dims, z, decode_weight, use_bias=True)
        KL_amounts.append(simple_KL)
        KL_names.append(f"simple_kl_{dims}")
        return x
    
    x = decode_latents_simple_kl(target_capacities, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names


def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    (Same as original - this part is fine)
    """
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs


def take_step_kl_only_simplified(task, model, optimizer, train_step, train_history_logger):
    """
    Training step that uses ORIGINAL decoding but ONLY replaces KL calculation.
    This is the proper null hypothesis test.
    """
    optimizer.zero_grad()
    
    # We need to replicate the forward pass but with our modified decode function
    # Get the necessary components from the model
    
    # Use our modified decode function that keeps everything the same except KL
    x, KL_amounts, KL_names = simple_decode_latents_kl_only(
        model.target_capacities, 
        model.decode_weights, 
        model.multiposteriors
    )
    
    # Continue with the ORIGINAL forward pass from here
    # (This is copied from the original ARCCompressor.forward())
    
    for layer_num in range(model.n_layers):
        # Multitensor communication layer
        x = layers.share_up(x, model.share_up_weights[layer_num])

        # Softmax layer
        x = layers.softmax(x, model.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

        # Directional layers
        x = layers.cummax(
            x, model.cummax_weights[layer_num], model.multitensor_system.task.masks,
            pre_norm=False, post_norm=True, use_bias=False
        )
        x = layers.shift(
            x, model.shift_weights[layer_num], model.multitensor_system.task.masks,
            pre_norm=False, post_norm=True, use_bias=False
        )

        # Directional communication layer
        x = layers.direction_share(x, model.direction_share_weights[layer_num], pre_norm=True, use_bias=False)

        # Nonlinear layer
        x = layers.nonlinear(x, model.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

        # Multitensor communication layer
        x = layers.share_down(x, model.share_down_weights[layer_num])

        # Normalization layer
        x = layers.normalize(x)

    # Linear Heads (ORIGINAL)
    logits = layers.affine(x, model.output_weights, use_bias=True)
    x_mask = layers.affine(x, model.x_mask_weights, use_bias=True)
    y_mask = layers.affine(x, model.y_mask_weights, use_bias=True)

    # ORIGINAL final processing
    logits = torch.cat([logits, torch.zeros_like(logits[:, :1, :, :])], dim=1)
    logits = torch.cat([torch.zeros_like(logits[:, :, :, :1]), logits], dim=3)
    logits = layers.position_mask(logits, model.multitensor_system.task.masks, model.masks_weights)

    x_mask = torch.cat([x_mask, torch.zeros_like(x_mask[:, :1, :])], dim=1)
    y_mask = torch.cat([y_mask, torch.zeros_like(y_mask[:, :, :1])], dim=2)

    output = torch.cat([logits, x_mask.unsqueeze(1), y_mask.unsqueeze(1)], dim=1)

    # Extract the logits and masks for training
    logits = output[:, :-2, :, :, :]
    x_mask = output[:, -2, :, :, :]
    y_mask = output[:, -1, :, :, :]

    # Compute total KL (using our simple KL amounts)
    total_KL = torch.tensor(0.0, requires_grad=True)
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)

    # Add black color to logits (same as original)
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)

    # Compute the reconstruction error (IDENTICAL to original)
    reconstruction_error = torch.tensor(0.0, requires_grad=True)
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
                
            logits_slice = logits[example_num,:,:,:,in_out_mode]
            problem_slice = task.problem[example_num,:,:,in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            
            x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')
                    logprobs[x_offset].append(logprob)
            
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)
            
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient
            reconstruction_error = reconstruction_error - logprob

    # Total loss (same weighting as original)
    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log the results
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)
