from typing import List, Tuple
import torch
import numpy as np

# --- Simplified MultiTensor System ---
# We'll simplify to 3 dimensions: [examples, colors, positions]
# This means there are 2^3 = 8 possible dimension combinations.


class SimpleMultiTensorSystem:
    """A simplified system for handling multi-dimensional configurations."""

    def __init__(self, n_examples, n_colors, n_positions):
        self.dim_lengths = [n_examples, n_colors, n_positions]
        self.num_dims = len(self.dim_lengths)
        print(f"Initialized SimpleMultiTensorSystem with dims: {self.dim_lengths}")

    def dims_valid(self, dims):
        """A simplified validity rule: at least one dimension must be active."""
        return sum(dims) > 0

    def shape(self, dims, channel_dim):
        """Creates a shape tuple based on active dimensions."""
        shape = [self.dim_lengths[i] for i, d in enumerate(dims) if d]
        shape.append(channel_dim)
        return tuple(shape)

    def __iter__(self):
        """Yields all valid dimension combinations."""
        for i in range(1, 2**self.num_dims):  # Skip the all-zero combination
            dims = [(i >> bit) & 1 for bit in range(self.num_dims)]
            if self.dims_valid(dims):
                yield dims

    def _make_multitensor(self, default, index):
        """Recursively creates the nested list structure for a MultiTensor."""
        if index == self.num_dims:
            return default
        return [self._make_multitensor(default, index + 1) for _ in range(2)]

    def make_multitensor(self, default=None):
        """Creates a MultiTensor with a default value at each leaf."""
        return SimpleMultiTensor(self._make_multitensor(default, 0), self)


class SimpleMultiTensor:
    """A simplified wrapper for a nested data structure."""

    def __init__(self, data, system):
        self.data = data
        self.system = system

    def __getitem__(self, dims):
        d = self.data
        for dim_val in dims:
            d = d[dim_val]
        return d

    def __setitem__(self, dims, value):
        d = self.data
        for dim_val in dims[:-1]:
            d = d[dim_val]
        d[dims[-1]] = value

    def __repr__(self):
        # A simple representation to see what's inside
        info = "SimpleMultiTensor holding:\n"
        for dims in self.system:
            tensor = self[dims]
            if tensor is not None:
                info += f"  - dims {dims}: Tensor of shape {tensor.shape}\n"
        return info


def multify(fn):
    """
    The decorator that applies a function to all valid dimension combinations
    if any arguments are MultiTensor instances.
    """

    def wrapper(*args, **kwargs):
        system = None
        # Check for MultiTensor instances in args and kwargs
        all_args = list(args) + list(kwargs.values())
        multi_tensors = [arg for arg in all_args if isinstance(arg, SimpleMultiTensor)]

        if not multi_tensors:
            # Normal mode: call the function directly, passing dims=None
            return fn(*args, **kwargs, dims=None)

        # Multi-mode: get the system and iterate
        system = multi_tensors[0].system
        result_mt = system.make_multitensor()

        for dims in system:
            # Unpack MultiTensors to get the specific tensor for the current dims
            new_args = [
                arg[dims] if isinstance(arg, SimpleMultiTensor) else arg for arg in args
            ]
            new_kwargs = {
                k: v[dims] if isinstance(v, SimpleMultiTensor) else v
                for k, v in kwargs.items()
            }

            # Call the original function with unpacked data and the current dims
            result_mt[dims] = fn(*new_args, **new_kwargs, dims=dims)

        return result_mt

    return wrapper


# --- Simplified Layers ---


@multify
def affine(x, weight, dims=None):
    """Simplified linear layer."""
    return torch.matmul(x, weight)


def channel_layer(posterior_mean):
    """
    A very simplified version of the VAE channel layer.
    It just adds noise and returns a dummy KL divergence value.
    """
    # "Sample" from the posterior using the reparameterization trick
    noise = torch.randn_like(posterior_mean)
    z = posterior_mean + noise * 0.1  # Add a small amount of noise

    # Return a dummy KL divergence value
    kl_divergence = torch.mean(posterior_mean**2)  # A mock KL value
    return z, kl_divergence


def decode_latents(posteriors, weights):
    """
    Simplified latent decoding. It's decorated, so it will run on each
    tensor within the `posteriors` and `weights` MultiTensors.
    """
    KL_amounts = {}

    @multify
    def _decode(posterior_mean, weight, dims=None):
        # 1. Sample from the latent space
        z, kl = channel_layer(posterior_mean)
        if dims:
            KL_amounts[tuple(dims)] = kl.item()

        # 2. Apply a linear layer
        return affine(z, weight)

    # The multify decorator handles the iteration
    x = _decode(posteriors, weights)
    return x, KL_amounts


@multify
def share_up(x, system, dims=None):
    """
    Simplified hierarchical communication.
    A tensor gets information from all tensors "below" it in the hierarchy.
    """
    total_from_lower_levels = 0
    current_tensor = x

    if dims is None:
        return current_tensor

    for lower_dims in system:
        # A tensor is "lower" if it's a strict subset of the current tensor's dims
        is_lower = sum(lower_dims) < sum(dims) and all(
            d_low <= d_high for d_low, d_high in zip(lower_dims, dims)
        )
        if is_lower:
            # In a real scenario, you'd access the tensor from the multitensor `x` itself.
            # This is a placeholder to simulate the effect of aggregating information.
            total_from_lower_levels += 0.1

    return current_tensor + total_from_lower_levels


# --- Simplified Model ---


class SimpleCompressor:
    """A simplified version of the ARCCompressor model."""

    def __init__(self, system):
        self.system = system
        self.channel_dim = 8  # Let's use a fixed channel dimension

        # 1. Initialize posteriors (just the means for simplicity)
        self.posteriors = self.system.make_multitensor()
        for dims in self.system:
            shape = self.system.shape(dims, self.channel_dim)
            # These are learnable parameters, so they need to require gradients.
            self.posteriors[dims] = torch.randn(shape, requires_grad=True)

        # 2. Initialize weights for the decoding layer
        self.decode_weights = self.system.make_multitensor()
        for dims in self.system:
            # These are also learnable parameters.
            shape = (self.channel_dim, self.channel_dim)
            self.decode_weights[dims] = torch.randn(shape, requires_grad=True)

    def parameters(self):
        """Collects all learnable parameters for the optimizer."""
        params = []
        for mt in [self.posteriors, self.decode_weights]:
            for dims in self.system:
                tensor = mt[dims]
                if tensor is not None and tensor.requires_grad:
                    params.append(tensor)
        return params

    def forward(self):
        """The simplified forward pass."""
        # 1. Decode latents from posteriors
        x, kl_info = decode_latents(self.posteriors, self.decode_weights)

        # 2. Hierarchical communication
        x = share_up(x, self.system)

        return x, kl_info


# --- Main Execution ---

if __name__ == "__main__":
    # Setup
    simple_system = SimpleMultiTensorSystem(n_examples=2, n_colors=4, n_positions=25)

    # Create the model
    model = SimpleCompressor(simple_system)

    # Create a dummy target for the model to learn
    target_output = simple_system.make_multitensor()
    for dims in simple_system:
        shape = simple_system.shape(dims, model.channel_dim)
        target_output[dims] = torch.rand(shape) * 5  # Some random target data

    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    print("\n--- STARTING TRAINING LOOP ---")
    for epoch in range(101):
        # Forward pass
        predicted_output, kl_info = model.forward()

        # --- Calculate loss ---
        # 1. Reconstruction Loss (how well the output matches the target)
        reconstruction_loss = 0
        for dims in simple_system:
            pred = predicted_output[dims]
            target = target_output[dims]
            reconstruction_loss += loss_fn(pred, target)

        # 2. KL Divergence Loss (a regularizer, from the VAE)
        kl_values = [torch.tensor(v) for v in kl_info.values()]
        kl_loss = sum(kl_values) * 0.01 if kl_values else torch.tensor(0.0)

        total_loss = reconstruction_loss + kl_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}, Total Loss: {total_loss.item():.4f} (Reconstruction: {reconstruction_loss.item():.4f}, KL: {kl_loss.item():.4f})"
            )

    print("--- TRAINING COMPLETE ---")

    # Let's see the final output
    final_output, _ = model.forward()
    print("\nFinal Output MultiTensor after training:")
    print(final_output)

    # We can inspect the final tensor for a specific, high-level combination
    final_grid_tensor = final_output[[1, 1, 1]]
    print("\nExample final tensor for dims [1,1,1] (examples, colors, positions):")
    print(f"Shape: {final_grid_tensor.shape}")


def load_examples() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (input_grid, output_grid) pairs as numpy int arrays."""
    # Example from the screenshot
    inp = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 2, 0, 2, 0],
            [2, 2, 2, 2, 2],
        ]
    )
    out = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0],
            [2, 1, 2, 1, 2],
        ]
    )

    inp2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [2, 2, 2, 2, 2],
        ]
    )

    out2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [2, 2, 1, 2, 2],
        ]
    )

    inp3 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 2, 0, 0, 2],
            [2, 2, 2, 2, 2],
        ]
    )

    out3 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2],
            [2, 1, 2, 2, 1],
        ]
    )

    test_inp = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [2, 0, 0, 0, 2],
            [2, 2, 2, 2, 2],
        ]
    )

    return [(inp, out), (inp2, out2), (inp3, out3)], [(test_inp, None)]
