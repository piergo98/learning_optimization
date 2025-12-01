
# Auto-generated convenience wrapper for function 'cost'
import torch
import gen_file as gen

def run_cost(inputs: torch.Tensor, device: str = 'cpu', dtype=torch.float32):
    """Run casadi->torch implementation for function 'cost'.

    Args:
        inputs: torch.Tensor with shape (B, total_nnz_in) containing concatenated inputs
        device: device string for allocated outputs/work
        dtype: torch dtype for allocated tensors

    Returns:
        torch.Tensor: output shaped (B, 1, 1)
    """
    _NNZ_IN = gen._cost_NNZ_IN
    _NNZ_OUT = gen._cost_NNZ_OUT
    _SZ_W = gen._cost_SZ_W

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

    # Split concatenated inputs into list of tensors matching nnz per input
    inputs_list = []
    offset = 0
    for nnz in _NNZ_IN:
        if nnz == 0:
            inputs_list.append(torch.empty((B, 0), device=device, dtype=dtype))
        else:
            inputs_list.append(inputs[:, offset:offset+nnz].contiguous())
        offset += nnz

    # Allocate outputs and work buffers expected by the generated function
    outputs = [torch.empty((B, n), device=device, dtype=dtype).contiguous() for n in _NNZ_OUT]
    work = torch.empty((B, _SZ_W), device=device, dtype=dtype)

    # Call generated low-level function (in-place on outputs/work)
    gen._cost(outputs, inputs_list, work)

    # Return first output reshaped to original output matrix shape
    out = outputs[0].reshape((B, 1, 1))
    return out


if __name__ == '__main__':
    # Quick smoke test (requires the user to provide sample inputs)
    print('Module gen_file wrapper for cost generated.')
