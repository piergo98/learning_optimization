
# Auto-generated convenience wrapper for function 'cost'
import torch
import gen_file as gen

def run_cost(inputs: torch.Tensor, device: str = 'cpu', dtype=torch.float32):
    """Run casadi->torch implementation for function 'cost'.

    Args:
        inputs: torch.Tensor with shape (B, total_nnz_in) containing concatenated inputs
        device: device string for computation
        dtype: torch dtype for computation

    Returns:
        torch.Tensor: output shaped (B, 1, 1)
    """
    _NNZ_IN = gen._cost_NNZ_IN

    if not torch.is_tensor(inputs):
        raise TypeError("inputs must be a torch.Tensor")

    if inputs.dim() < 2:
        raise ValueError("inputs must have shape (B, total_nnz_in)")

    B = inputs.shape[0]
    total_nnz = sum(_NNZ_IN)
    if inputs.shape[1] != total_nnz:
        raise ValueError(f"Expected input width {total_nnz}, got {inputs.shape[1]}")

    # Ensure correct device/dtype and contiguous memory once
    inputs = inputs.to(device=device, dtype=dtype).contiguous()

    # Call autograd-compatible generated function (no in-place operations)
    output = gen.cost(inputs, device=device, dtype=dtype)

    return output


if __name__ == '__main__':
    # Quick smoke test (requires the user to provide sample inputs)
    total_nnz_in = sum(gen._cost_NNZ_IN)
    batch_size = 10
    inputs = torch.randn(batch_size, total_nnz_in)
    print(f"Running gen_file_torch.run_cost with random input of shape {inputs.shape}")
    output = run_cost(inputs)
    print(f"Output shape: {output.shape}")
    print('Module gen_file wrapper for cost generated.')
