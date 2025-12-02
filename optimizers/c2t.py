import os
import textwrap
import argparse
import casadi
from casadi import *
from cusadiops import *
from tqdm import tqdm
import shutil

def modified_generate_pytorch_code(f: casadi.Function):
    """
    Generates PyTorch code for a CasADi function by analyzing its computational graph
    and mapping each operation to its PyTorch equivalent.
    
    Args:
        f (casadi.Function): CasADi function object loaded from .casadi file
        
    Returns:
        str: Complete Python function definition with metadata
        
    Notes:
        - Generated functions follow naming convention: _{original_name}
        - Function signature: func(outputs, inputs, work) for in-place computation
        - Memory layout matches CasADi's graph
        
    Raises:
        NotImplementedError: If CasADi operation has no PyTorch equivalent
    """
    f_name = f.name()

    # ==================== CasADi Function Analysis ====================
    
    # Core function properties
    n_instr = f.n_instructions()   # Number of elementary operations in the graph
    n_in = f.n_in()                # Number of input arguments
    n_out = f.n_out()              # Number of output arguments  
    n_w = f.sz_w()                 # Size of work vector (intermediate storage)

    # Each instruction represents one elementary operation (add, mul, sin, etc.)
    input_idx = [f.instruction_input(i) for i in range(n_instr)]      # Input indices for each op
    output_idx = [f.instruction_output(i) for i in range(n_instr)]    # Output indices for each op
    operations = [f.instruction_id(i) for i in range(n_instr)]        # Operation type IDs
    const_instr = [f.instruction_constant(i) for i in range(n_instr)] # Constant values

    # Extract input and output names for metadata
    input_names = [f.name_in(i) for i in range(n_in)]    # Symbolic names (e.g., 'q', 'dq', 'tau')
    output_names = [f.name_out(i) for i in range(n_out)] # Output names (e.g., 'ddq', 'tau_ext')

    # ==================== Code Generation ====================
    # We will append '_' to all variables and functions to avoid conflicts in the wrapper
    n_in_str = f"_{f_name}_N_IN = {n_in}\n"
    n_out_str = f"_{f_name}_N_OUT = {n_out}\n"
    n_instr_str = f"_{f_name}_N_INSTR = {n_instr}\n"
    
    # Sparsity pattern information (number of non-zeros)
    nnz_in_str = f"_{f_name}_NNZ_IN = {[f.nnz_in(i) for i in range(n_in)]}\n"
    nnz_out_str = f"_{f_name}_NNZ_OUT = {[f.nnz_out(i) for i in range(n_out)]}\n"
    
    input_names_str = f"_{f_name}_INPUT_NAMES = {input_names}\n"
    output_names_str = f"_{f_name}_OUTPUT_NAMES = {output_names}\n"

    n_w_str = f"_{f_name}_SZ_W = {n_w}\n"

    # Combine all metadata into a preamble
    preamble_variables = n_in_str + n_out_str + n_instr_str + n_w_str + nnz_in_str + nnz_out_str + input_names_str + output_names_str

    # Generate function.
    # TODO: in theory we could use torch.compile or torch.jit, but these functions may easily
    # get way too large (hundreds of thoudsands of operations) so sadly it's not a good idea.
    # TODO: might be wise to check if we can wrap these functios as torch.nn.Module
    function_signature = f"def _{f_name}(outputs, inputs, work):"
    str_operations = preamble_variables + function_signature

    # ==================== Operation Translation ====================
    # Convert each CasADi instruction to equivalent PyTorch operation
    # This basically comes from Cusadi.
    
    for k in range(n_instr):
        op = operations[k]             # Current operation type
        o_idx = output_idx[k]          # Where to store result
        i_idx = input_idx[k]           # Input operand indices
        
        if op == OP_CONST:
            # Load constant value: work[o_idx] = constant
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], const_instr[k])
            
        elif op == OP_INPUT:
            # Copy from input buffer: work[o_idx] = inputs[input_idx][element_idx]
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
            
        elif op == OP_OUTPUT:
            # Copy to output buffer: outputs[output_idx][element_idx] = work[i_idx]
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], o_idx[1], i_idx[0])
            
        elif op == OP_SQ:
            # Square operation: work[o_idx] = work[i_idx] * work[i_idx]
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[0])
            
        # Generic operation handling based on argument count
        elif OP_PYTORCH_DICT[op].count("%d") == 3:
            # Binary operations: work[o_idx] = work[i_idx[0]] op work[i_idx[1]]
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
            
        elif OP_PYTORCH_DICT[op].count("%d") == 2:
            # Unary operations: work[o_idx] = op(work[i_idx[0]])
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0])
            
        else:
            # Operation not yet implemented in OP_PYTORCH_DICT
            raise NotImplementedError(
                f"CasADi operation '{op}' is not supported. "
                f"Add mapping to OP_PYTORCH_DICT in cusadiops.py"
            )

    return str_operations + "\n"


def c2t(cusadi_folder="casadi_functions", output_dir=None, output_file="gen_file.py"):
    """
    Batch processes all .casadi files in a directory and generates a unified PyTorch module.
    
    Args:
        cusadi_folder (str): Path to directory containing .casadi files
                           Expected structure: robot_dir/casadi_functions/*.casadi
        output_dir (str, optional): Target directory for generated code
                                  Creates directory if it doesn't exist
        output_file (str): Name of generated Python file (default: "gen_file.py")
        
    """

    if not os.path.exists(cusadi_folder):
        print(f"Error: The folder '{cusadi_folder}' was not found.")
        print(f"Expected structure: robot_dir/casadi_functions/*.casadi")
        return

    # Discover all CasADi function files
    cusadi_files = [f for f in os.listdir(cusadi_folder) if f.endswith(".casadi")]

    if not cusadi_files:
        print(f"No .casadi files were found in '{cusadi_folder}'.")
        print(f"Ensure CasADi functions have been exported to this directory.")
        return

    print(f"Found {len(cusadi_files)} .casadi files to process:")
    for file in cusadi_files:
        print(f"  - {file}")

    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        print(f"Output directory: {output_dir}")
    else:
        output_path = output_file
        print(f"Output to current directory: {os.getcwd()}")

    # ==================== Code Generation ====================
    # Initialize
    all_generated_code = textwrap.dedent('''
    # ! AUTOMATICALLY GENERATED CODE
    # Generated by SwiLin PyTorch Code Generator
    # Source: CasADi functions from robotics dynamics computations
    # 
    # WARNING: Do not edit manually - changes will be overwritten
    # Regenerate using: python thunder_torch_gen.py --robot_dir <path>
    
    import torch

    ''')

    # Process each .casadi file and accumulate generated code
    successful_conversions = 0
    failed_conversions = []

    # Track metadata for wrapper generation
    functions_meta = []  # list of dicts: name, nnz_in, nnz_out, input_names, output_names, sz_w

    pbar = tqdm(cusadi_files, desc="Processing files", unit="file")
    for cusadi_file in pbar:
        try:
            func_path = os.path.join(cusadi_folder, cusadi_file)

            f = Function.load(func_path)
            
            generated_code = modified_generate_pytorch_code(f)
            
            all_generated_code += generated_code

            #function name is 'get_' + cusadi_file_name without .casadi
            function_name = os.path.splitext(cusadi_file)[0]

            # get output shape
            output_shape = (f.sparsity_out(0).size1(), f.sparsity_out(0).size2())
            # stash meta used for wrapper
            functions_meta.append({
                "name": "get_" + function_name,
                "internal_name": f.name(),
                "nnz_in": [f.nnz_in(i) for i in range(f.n_in())],
                "nnz_out": [f.nnz_out(i) for i in range(f.n_out())],
                "input_names": [f.name_in(i) for i in range(f.n_in())],
                "output_names": [f.name_out(i) for i in range(f.n_out())],
                "sz_w": f.sz_w(),
                "output_shape": output_shape,
            })
            
            successful_conversions += 1
            
        except Exception as e:
            failed_conversions.append((cusadi_file, str(e)))
            
            all_generated_code += f"\n# ERROR: Could not convert {cusadi_file}\n"
            all_generated_code += f"# Reason: {e}\n\n"

    # Write all the generated code to the output file
    try:
        with open(output_path, "w") as out_f:
            out_f.write(all_generated_code)
        
        print(f"\n{'='*60}")
        print(f"Code generation completed!")
        print(f"Output file: {output_path}")
        print(f"Successfully converted: {successful_conversions}/{len(cusadi_files)} functions")
        
        if failed_conversions:
            print(f"\nFailed conversions:")
            for file, error in failed_conversions:
                print(f"  - {file}: {error}")
                
    except Exception as e:
        print(f"\nFATAL ERROR: Could not write output file '{output_path}': {e}")
        return

    print(f"{'='*60}")

    # ==================== Single-function Torch wrapper generation ====================
    # Create a simple, single-function PyTorch wrapper for the first converted CasADi function.
    # If multiple functions were converted, the first one is used (print a warning).
    if not functions_meta:
        print("No functions metadata available; skipping single-function wrapper generation.")
    else:
        if len(functions_meta) > 1:
            print("Warning: multiple functions converted — creating a wrapper for the first one only.")

        meta = functions_meta[0]
        iname = meta["internal_name"]     # CasADi internal name (used in generated variables / function)
        fname = meta["name"]              # friendly name (e.g. get_<file>)
        output_shape = meta["output_shape"]

        # Module that was just generated above
        module_basename = os.path.splitext(os.path.basename(output_path))[0]
        wrapper_filename = f"{module_basename}_torch.py"
        wrapper_path = os.path.join(output_dir if output_dir else os.getcwd(), wrapper_filename)

        # Build the wrapper source: imports + a convenience function run_<fname>
        # The wrapper now accepts a single concatenated torch.Tensor of shape (B, total_nnz_in)
        wrapper_code = textwrap.dedent(f"""
        # Auto-generated convenience wrapper for function '{iname}'
        import torch
        import {module_basename} as gen

        def run_{iname}(inputs: torch.Tensor, device: str = 'cpu', dtype=torch.float32):
            \"\"\"Run casadi->torch implementation for function '{iname}'.

            Args:
                inputs: torch.Tensor with shape (B, total_nnz_in) containing concatenated inputs
                device: device string for allocated outputs/work
                dtype: torch dtype for allocated tensors

            Returns:
                torch.Tensor: output shaped (B, {output_shape[0]}, {output_shape[1]})
            \"\"\"
            _NNZ_IN = gen._{iname}_NNZ_IN
            _NNZ_OUT = gen._{iname}_NNZ_OUT
            _SZ_W = gen._{iname}_SZ_W

            if not torch.is_tensor(inputs):
                raise TypeError(\"inputs must be a torch.Tensor\")

            if inputs.dim() < 2:
                raise ValueError(\"inputs must have shape (B, total_nnz_in)\")

            B = inputs.shape[0]
            total_nnz = sum(_NNZ_IN)
            if inputs.shape[1] != total_nnz:
                raise ValueError(f\"Expected input width {{total_nnz}}, got {{inputs.shape[1]}}\")

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
            gen._{iname}(outputs, inputs_list, work)

           
            # Replace NaNs with zeros (non-in-place to preserve autograd graph)
            out = torch.where(torch.isnan(outputs[0]), torch.tensor(0.0, device=device, dtype=dtype), outputs[0])ut


        if __name__ == '__main__':
            # Quick smoke test (requires the usero provout inputs)
            total_nnz_in = sum(gen._cost_Nndn(1, total_nnz_in)
            print(f"Running gen_file_torch.run_cost with random input of shape {{inputs.shape}}")
            output = run_cost(inputs)
            print("Output of run_cost:")
            print(output)
            print(f"Output shape: {{output.shape}}")
            print('Module {module_basename} wrapper for {iname} generated.')
        """)

        try:
            os.makedirs(os.path.dirname(wrapper_path) or ".", exist_ok=True)
            with open(wrapper_path, "w") as wf:
                wf.write(wrapper_code)
            print(f"Single-function torch wrapper generated: {wrapper_path}")
        except Exception as e:
            print(f"ERROR: Could not write single-function wrapper '{wrapper_path}': {e}")

    # # ==================== Single-function Torch wrapper generation ====================
    # # Create a simple, single-function PyTorch wrapper for the first converted CasADi function.
    # # If multiple functions were converted, the first one is used (print a warning).
    # if not functions_meta:
    #     print("No functions metadata available; skipping single-function wrapper generation.")
    # else:
    #     if len(functions_meta) > 1:
    #         print("Warning: multiple functions converted — creating a wrapper for the first one only.")

    #     meta = functions_meta[0]
    #     iname = meta["internal_name"]     # CasADi internal name (used in generated variables / function)
    #     fname = meta["name"]              # friendly name (e.g. get_<file>)
    #     output_shape = meta["output_shape"]

    #     # Module that was just generated above
    #     module_basename = os.path.splitext(os.path.basename(output_path))[0]
    #     wrapper_filename = f"{module_basename}_torch.py"
    #     wrapper_path = os.path.join(output_dir if output_dir else os.getcwd(), wrapper_filename)

    #     # Build the wrapper source: imports + a convenience function run_<fname>
    #     # The wrapper expects a dict mapping input names -> torch.Tensor of shape (B, nnz_in)
    #     wrapper_code = textwrap.dedent(f"""
    #     # Auto-generated convenience wrapper for function '{iname}'
    #     import torch
    #     import {module_basename} as gen

    #     def run_{iname}(inputs: dict, device: str = 'cpu', dtype=torch.float32):
    #         \"\"\"Run casadi->torch implementation for function '{iname}'.

    #         Args:
    #             inputs: dict mapping input_name (str) -> torch.Tensor with shape (B, nnz_in)
    #             device: device string for allocated outputs/work
    #             dtype: torch dtype for allocated tensors

    #         Returns:
    #             torch.Tensor: output shaped (B, {output_shape[0]}, {output_shape[1]})
    #         \"\"\"
    #         # Read metadata from generated module
    #         _INPUT_NAMES = gen._{iname}_INPUT_NAMES
    #         _NNZ_OUT = gen._{iname}_NNZ_OUT
    #         _SZ_W = gen._{iname}_SZ_W

    #         # Build ordered input list from provided dict
    #         inputs_list = []
    #         B = None
    #         for nm in _INPUT_NAMES:
    #             if nm not in inputs:
    #                 raise KeyError(f\"Missing input '{{nm}}' required by function '{iname}'\")
    #             t = inputs[nm]
    #             if not torch.is_tensor(t):
    #                 raise TypeError(f\"Input '{{nm}}' must be a torch.Tensor\")
    #             if B is None:
    #                 if t.dim() < 1:
    #                     raise ValueError(f\"Input '{{nm}}' must have a batch dimension\")
    #                 B = t.shape[0]
    #             # ensure contiguous on the chosen device/dtype
    #             if t.device.type != device:
    #                 t = t.to(device=device)
    #             if t.dtype != dtype:
    #                 t = t.to(dtype=dtype)
    #             inputs_list.append(t.contiguous())

    #         if B is None:
    #             # no inputs provided (degenerate) -> assume batch 1
    #             B = 1

    #         # Allocate outputs and work buffers expected by the generated function
    #         outputs = [torch.empty((B, n), device=device, dtype=dtype).contiguous() for n in _NNZ_OUT]
    #         work = torch.empty((B, _SZ_W), device=device, dtype=dtype)

    #         # Call generated low-level function (in-place on outputs/work)
    #         gen._{iname}(outputs, inputs_list, work)

    #         # Return first output reshaped to original output matrix shape
    #         out = outputs[0].reshape((B, {output_shape[0]}, {output_shape[1]}))
    #         return out


    #     if __name__ == '__main__':
    #         # Quick smoke test (requires the user to provide sample inputs)
    #         print('Module {module_basename} wrapper for {iname} generated.')
    #     """)

    #     try:
    #         os.makedirs(os.path.dirname(wrapper_path) or ".", exist_ok=True)
    #         with open(wrapper_path, "w") as wf:
    #             wf.write(wrapper_code)
    #         print(f"Single-function torch wrapper generated: {wrapper_path}")
    #     except Exception as e:
    #         print(f"ERROR: Could not write single-function wrapper '{wrapper_path}': {e}")
            

if __name__ == '__main__':
    """
    Command-line interface for PyTorch code generator.
    
    
    Usage Examples:
        # Looks for casadi_functions/ subfolder
        python c2t.py --func_dir /path/to/franka_generatedFiles
        
        # Full specification with custom output
        python c2t.py \\
            --func_dir /path/to/franka_generatedFiles \\
            --output_dir /path/to/torch_lib \\
    
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--func_dir', type=str, 
                       help='Path to robot generated files directory containing casadi_functions/ folder')
    parser.add_argument('--output_dir', type=str,
                       help='Path to output directory for generated PyTorch library')
    
    args = parser.parse_args()
    
    # ==================== Path Resolution ====================
    if args.func_dir:
        # Look for casadi_functions folder in the robot directory
        # This follows the convention: casadi_functions/*.casadi
        cusadi_folder = os.path.join(args.func_dir, "casadi_functions")
        print(f"Using directory: {args.func_dir}")
        print(f"Looking for CasADi functions in: {cusadi_folder}")
    else:
        cusadi_folder = "casadi_functions"
        print(f"Using default CasADi functions directory: {cusadi_folder}")
    
    # ==================== Execute Code Generation ====================
    c2t(cusadi_folder, args.output_dir)
