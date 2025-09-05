import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from simplified_arc import SimpleMultiTensorSystem, SimpleCompressor, load_examples

def numpy_to_torch(np_array, device='cpu'):
    """Convert a numpy array to a torch tensor."""
    return torch.from_numpy(np_array).float().to(device)

def create_target_multitensor(system, examples, channel_dim=8):
    """Create a target multitensor from ARC examples."""
    # Get the dimensions from the examples
    example_shape = examples[0][0].shape
    n_positions = example_shape[0] * example_shape[1]  # Total grid cells
    
    # Create the target multitensor
    target_output = system.make_multitensor()
    
    # For each valid dimension combination
    for dims in system:
        shape = system.shape(dims, channel_dim)
        # Initialize with zeros
        target_output[dims] = torch.zeros(shape)
        
        # If the example dimension is active, fill with actual data
        if dims[0] == 1:
            # Create one-hot encodings for each example
            for i, (inp, out) in enumerate(examples):
                if i >= shape[0]:  # Skip if we have more examples than the shape allows
                    break
                    
                # Flatten the grid and one-hot encode
                if out is not None:
                    flat_out = out.flatten()
                    for pos in range(n_positions):
                        color = int(flat_out[pos])
                        if dims[1] == 1:  # If colors dimension is active
                            if dims[2] == 1:  # If positions dimension is active
                                target_output[dims][i, color, pos, :] = 1.0
                            else:  # Positions dimension not active
                                target_output[dims][i, color, :] = 1.0
                        elif dims[2] == 1:  # Colors not active, but positions active
                            target_output[dims][i, pos, :] = 1.0 if color > 0 else 0.0
    
    return target_output

def save_prediction(prediction, example_idx, epoch, save_dir):
    """Save a visual representation of the model's prediction."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a 5x5 grid visualization from the prediction tensor
    grid = np.zeros((5, 5), dtype=np.int32)
    
    # Get the highest probability color for each position
    for pos in range(25):
        row, col = pos // 5, pos % 5
        probs = [prediction[example_idx, color, pos, 0].item() for color in range(4)]
        best_color = np.argmax(probs)
        grid[row, col] = best_color
    
    # Create a colored visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='viridis')
    plt.grid(True, color='black', linestyle='-', linewidth=1)
    plt.title(f'Prediction at Epoch {epoch}')
    plt.savefig(os.path.join(save_dir, f'prediction_example_{example_idx}_epoch_{epoch}.png'))
    plt.close()
    
    return grid

def main():
    # Load examples
    train_examples, test_examples = load_examples()
    
    # Setup
    n_examples = len(train_examples) + len(test_examples)
    n_colors = 4  # Based on the dataset (0, 1, 2)
    n_positions = 25  # 5x5 grid
    channel_dim = 8  # Same as in the original code
    
    # Create the system and model
    system = SimpleMultiTensorSystem(n_examples=n_examples, n_colors=n_colors, n_positions=n_positions)
    model = SimpleCompressor(system)
    
    # Create target output from examples
    target_output = create_target_multitensor(system, train_examples + test_examples, channel_dim)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    # Create directories for saving predictions
    save_dir = Path('arc_predictions')
    save_dir.mkdir(exist_ok=True)
    
    # Lists to store loss and predictions for plotting
    losses = []
    all_predictions = []
    
    print("\n--- STARTING TRAINING LOOP ---")
    for epoch in range(10001):  # Increase epochs to 200
        # Forward pass
        predicted_output, kl_info = model.forward()
        
        # --- Calculate loss ---
        # 1. Reconstruction Loss (how well the output matches the target)
        reconstruction_loss = 0
        for dims in system:
            pred = predicted_output[dims]
            target = target_output[dims]
            reconstruction_loss += loss_fn(pred, target)
        
        # 2. KL Divergence Loss (a regularizer, from the VAE)
        kl_values = [torch.tensor(v) for v in kl_info.values()]
        kl_loss = sum(kl_values) * 0.01 if kl_values else torch.tensor(0.0)
        
        total_loss = reconstruction_loss + kl_loss
        losses.append(float(total_loss))
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Save prediction for visualization
        # We'll focus on the most interpretable tensor: [examples, colors, positions]
        if [1, 1, 1] in system:
            prediction_tensor = predicted_output[[1, 1, 1]].detach().clone()
            
            # Save predictions for all examples
            for ex_idx in range(len(train_examples) + len(test_examples)):
                if epoch % 10 == 0 or epoch == 200:  # Save every 10 epochs and the final one
                    grid = save_prediction(
                        prediction_tensor, 
                        ex_idx, 
                        epoch, 
                        save_dir / f'example_{ex_idx}'
                    )
                    if epoch == 10000:  # Save the final prediction separately
                        all_predictions.append(grid)
        
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}, Total Loss: {float(total_loss):.4f} "
                f"(Reconstruction: {float(reconstruction_loss):.4f}, KL: {float(kl_loss):.4f})"
            )
    
    print("--- TRAINING COMPLETE ---")
    
    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_dir / 'loss_curve.png')
    
    # Final comparison: side by side visualization of input, target, and prediction
    fig, axes = plt.subplots(len(train_examples) + len(test_examples), 3, figsize=(15, 5 * (len(train_examples) + len(test_examples))))
    
    for i, examples in enumerate([train_examples, test_examples]):
        for j, (inp, out) in enumerate(examples):
            idx = i * len(train_examples) + j
            
            # Input
            axes[idx, 0].imshow(inp, cmap='viridis')
            axes[idx, 0].set_title(f'Input #{idx+1}')
            axes[idx, 0].grid(True, color='black', linestyle='-', linewidth=1)
            
            # Target (if available)
            if out is not None:
                axes[idx, 1].imshow(out, cmap='viridis')
                axes[idx, 1].set_title(f'Target #{idx+1}')
            else:
                axes[idx, 1].imshow(np.zeros_like(inp), cmap='viridis')
                axes[idx, 1].set_title(f'Target #{idx+1} (None)')
            axes[idx, 1].grid(True, color='black', linestyle='-', linewidth=1)
            
            # Prediction
            if idx < len(all_predictions):
                axes[idx, 2].imshow(all_predictions[idx], cmap='viridis')
                axes[idx, 2].set_title(f'Prediction #{idx+1}')
                axes[idx, 2].grid(True, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'final_comparison.png')
    
    print(f"Results saved in {save_dir.absolute()}")

if __name__ == "__main__":
    main()
