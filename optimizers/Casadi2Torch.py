import torch
import torch.nn as nn
import casadi as ca
import numpy as np

class CasadiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, casadi_fn, *inputs):
        """
        Generic forward pass for any CasADi function.
        inputs: Sequence of PyTorch tensors matching CasADi arguments.
        """
        # 1. Check dimensions and retrieve batch size
        # We assume the first dimension is the Batch dimension [Batch, Features]
        batch_size = inputs[0].shape[0]
        
        # 2. Prepare inputs for CasADi (Batch, Features) -> (Features, Batch)
        # CasADi .map() expects column-major data
        inputs_np = [x.detach().cpu().numpy().T for x in inputs]
        
        # 3. Run CasADi function
        # .map allows us to run the function over the whole batch at once
        result_dm = casadi_fn.map(batch_size)(*inputs_np)
        
        # 4. Handle output (CasADi might return a single item or a list)
        if isinstance(result_dm, (list, tuple)):
            results_np = [np.array(r).T for r in result_dm]
        else:
            results_np = [np.array(result_dm).T]
            
        # 5. Save context for backward (gradients)
        ctx.casadi_fn = casadi_fn
        ctx.save_for_backward(*inputs)
        ctx.batch_size = batch_size
        ctx.n_outputs = len(results_np)

        # 6. Convert back to torch
        results_torch = [torch.from_numpy(r).float().to(inputs[0].device) for r in results_np]
        
        # If specific function has only 1 output, unpack it
        return results_torch[0] if len(results_torch) == 1 else tuple(results_torch)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Generic backward pass using CasADi AD.
        """
        inputs = ctx.saved_tensors
        casadi_fn = ctx.casadi_fn
        batch_size = ctx.batch_size
        
        # 1. Prepare Inputs and Incoming Gradients (transposed)
        inputs_np = [x.detach().cpu().numpy().T for x in inputs]
        grad_outputs_np = [g.detach().cpu().numpy().T for g in grad_outputs]
        
        # 2. Compute Jacobian-Vector Product (Reverse Mode AD)
        # We assume the user wants gradients w.r.t all inputs
        # We must generate the reverse-mode AD function for THIS specific function structure
        
        # NOTE: For maximum efficiency, this graph generation should be cached in __init__, 
        # but for generic "conversion" we do it here safely.
        
        # Reconstruct symbolic inputs based on numerical shapes
        sym_inputs = [ca.MX.sym(f'in_{i}', inp.shape[0], 1) for i, inp in enumerate(inputs_np)]
        sym_out = casadi_fn(*sym_inputs)
        
        # Ensure sym_out is a list
        if not isinstance(sym_out, (list, tuple)):
            sym_out = [sym_out]

        # Calculate gradients: sum( adj * output )
        # CasADi requires matching the number of output gradients
        adj_inputs = ca.jtimes(ca.vertcat(*sym_out), ca.vertcat(*sym_inputs), ca.vertcat(*grad_outputs_np), True)
        
        # The result of jtimes is one giant vector; we need to split it back into input shapes
        # This split logic can be complex; a simpler approach for a quick converter 
        # is to ask CasADi for gradients one by one or create a specific function.
        
        # --- Optimized Reverse Mode Mapping ---
        # We create a function: f_bwd(inputs, grad_outputs) -> grad_inputs
        bwd_name = f"{casadi_fn.name()}_bwd"
        sym_grads = [ca.MX.sym(f'g_{i}', g.shape[0], 1) for i, g in enumerate(grad_outputs_np)]
        
        # Reverse mode AD
        # gradient of (outputs dot grad_outputs) w.r.t inputs
        out_dot_grad = 0
        for out_node, grad_node in zip(sym_out, sym_grads):
            out_dot_grad += ca.dot(out_node, grad_node)
            
        grads_per_input = ca.gradient(out_dot_grad, ca.vertcat(*sym_inputs))
        
        # Create the backward function
        # Inputs: [*inputs, *grad_outputs] -> Output: [grads_per_input]
        # Note: grads_per_input is a flat vector, might need splitting if inputs were distinct
        # For simplicity in this generic converter, we assume inputs are vector-like enough to be vertcat'd
        
        bwd_fn = ca.Function(bwd_name, [*sym_inputs, *sym_grads], [grads_per_input])
        
        # Execute
        all_grads_dm = bwd_fn.map(batch_size)(*inputs_np, *grad_outputs_np)
        all_grads_np = np.array(all_grads_dm) # Shape: (Total_Input_Dim, Batch)
        
        # Split gradients back to match input tensors
        grad_inputs_torch = []
        idx = 0
        for inp in inputs_np:
            rows = inp.shape[0]
            grad_segment = all_grads_np[idx : idx+rows, :]
            grad_inputs_torch.append(torch.from_numpy(grad_segment.T).float().to(inputs[0].device))
            idx += rows
            
        return (None, *grad_inputs_torch)

class CasadiToTorch(nn.Module):
    def __init__(self, casadi_fn):
        super().__init__()
        self.casadi_fn = casadi_fn
        
    def forward(self, *args):
        # The apply method handles the autograd linking
        return CasadiFunction.apply(self.casadi_fn, *args)