"""
Switched System Simulator

This module provides a class for simulating the evolution of switched linear/nonlinear systems
with user-defined switching conditions and control inputs.

Features:
- Continuous-time state evolution with adaptive ODE solving
- Flexible switching condition evaluation (state-based, time-based, external)
- Support for controlled and autonomous systems
- Trajectory logging and visualization
- Compatible with both numpy and PyTorch
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings
from scipy.integrate import solve_ivp
from scipy.special import softmax
import matplotlib.pyplot as plt
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Simulation will use NumPy only.")
    
from src.training import SwiLinNN


class SwitchedSystemSimulator:
    """
    Simulator for switched dynamical systems with arbitrary switching conditions.
    
    The simulator evolves the system state x(t) according to:
        dx/dt = A[σ(t)] @ x + B[σ(t)] @ u(t) + f[σ(t)](x, u, t)
    
    where σ(t) is the active mode determined by switching conditions.
    
    Supports three simulation modes:
    1. Event-driven switching (via `simulate()`) with dynamic switching conditions
    2. Predefined sequences (via `simulate_sequence()`) with fixed mode durations and controls
    3. Model Predictive Control (via `simulate_mpc()`) with receding horizon optimization
    
    Parameters
    ----------
    A : list of np.ndarray
        List of system matrices for each mode, shape (n_states, n_states) each.
    B : list of np.ndarray, optional
        List of input matrices for each mode, shape (n_states, n_inputs) each.
        If None, autonomous system is assumed.
    f : list of callable, optional
        List of nonlinear functions for each mode: f[i](x, u, t) -> np.ndarray.
        If None, linear dynamics are assumed.
    switching_conditions : list of callable, optional
        List of functions that determine when to switch from mode i.
        Each function has signature: condition(x, t, mode) -> bool.
        Returns True when system should switch away from current mode.
    switching_map : callable, optional
        Function that determines next mode: next_mode = switching_map(x, t, current_mode).
        If None, cycles through modes sequentially.
    mode_names : list of str, optional
        Names for each mode (for logging/visualization).
    
    Examples
    --------
    >>> # Example 1: Event-driven switching
    >>> A = [np.array([[-1, 0], [0, -2]]), np.array([[-2, 1], [-1, -1]])]
    >>> B = [np.array([[1], [0]]), np.array([[0], [1]])]
    >>> def switch_condition(x, t, mode):
    ...     return np.linalg.norm(x) < 0.5
    >>> sim = SwitchedSystemSimulator(A, B, switching_conditions=[switch_condition])
    >>> result = sim.simulate(x0, t_span=(0, 10), control=lambda t: np.array([0.1]))
    >>>
    >>> # Example 2: Predefined sequence with piecewise constant controls
    >>> mode_sequence = [0, 1, 0]  # mode indices
    >>> durations = [2.0, 3.0, 1.5]  # time spent in each mode
    >>> controls = [np.array([0.5]), np.array([0.2]), np.array([0.8])]  # constant control per phase
    >>> result = sim.simulate_sequence(x0, mode_sequence, durations, controls)
    """
    
    def __init__(
        self,
        A: List[np.ndarray],
        B: Optional[List[np.ndarray]] = None,
        f: Optional[List[Callable]] = None,
        switching_conditions: Optional[List[Callable]] = None,
        switching_map: Optional[Callable] = None,
        mode_names: Optional[List[str]] = None,
        enforce_continuous_control: bool = False,
        dtype: Optional[type] = None,
    ):
        
        self.dtype = dtype if dtype is not None else np.float32
        # Validate and store system matrices
        self.n_modes = len(A)
        if self.n_modes == 0:
            raise ValueError("At least one mode must be provided.")
        
        # Helper to robustly convert inputs (including torch tensors on CUDA) to numpy
        def _to_numpy(arr):
            if TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            return np.asarray(arr, dtype=self.dtype)

        self.A = [_to_numpy(a) for a in A]
        self.n_states = self.A[0].shape[0]
        
        # Validate all A matrices have consistent dimensions
        for i, a in enumerate(self.A):
            if a.shape != (self.n_states, self.n_states):
                raise ValueError(f"A[{i}] has shape {a.shape}, expected ({self.n_states}, {self.n_states})")
        
        # Handle input matrices
        if B is None:
            # Autonomous system
            self.B = [np.zeros((self.n_states, 0)) for _ in range(self.n_modes)]
            self.n_inputs = 0
            self.autonomous = True
        else:
            self.B = [_to_numpy(b) for b in B]
            self.n_inputs = self.B[0].shape[1]
            self.autonomous = False
            
            # Validate B matrices
            for i, b in enumerate(self.B):
                if b.shape[0] != self.n_states:
                    raise ValueError(f"B[{i}] has {b.shape[0]} rows, expected {self.n_states}")
                if b.shape[1] != self.n_inputs:
                    raise ValueError(f"B[{i}] has {b.shape[1]} columns, expected {self.n_inputs}")
        
        # Store nonlinear functions (default to zero)
        if f is None:
            self.f = [lambda x, u, t: np.zeros(self.n_states) for _ in range(self.n_modes)]
        else:
            if len(f) != self.n_modes:
                raise ValueError(f"Expected {self.n_modes} nonlinear functions, got {len(f)}")
            self.f = f
        
        # Store switching conditions
        self.switching_conditions = switching_conditions
        self.switching_map = switching_map
        
        # Mode names for logging
        if mode_names is None:
            self.mode_names = [f"Mode {i}" for i in range(self.n_modes)]
        else:
            if len(mode_names) != self.n_modes:
                raise ValueError(f"Expected {self.n_modes} mode names, got {len(mode_names)}")
            self.mode_names = mode_names
            
        self.mode_seq = [i for i in range(self.n_modes)]
        
        self.enforce_continuous_control = enforce_continuous_control
        
        # History storage
        self.reset_history()
    
    def reset_history(self):
        """Clear simulation history."""
        self.t_history = []
        self.x_history = []
        self.mode_history = []
        self.switch_times = []
        self.u_history = []
    
    def _check_switching_condition(self, x: np.ndarray, t: float, mode: int) -> bool:
        """
        Check if system should switch from current mode.
        
        Parameters
        ----------
        x : np.ndarray
            Current state.
        t : float
            Current time.
        mode : int
            Current active mode.
            
        Returns
        -------
        bool
            True if switching condition is satisfied.
        """
        if self.switching_conditions is None:
            return False
        
        if mode >= len(self.switching_conditions):
            return False
        
        condition = self.switching_conditions[mode]
        if condition is None:
            return False
        
        try:
            return condition(x, t, mode)
        except Exception as e:
            warnings.warn(f"Switching condition evaluation failed at t={t}, mode={mode}: {e}")
            return False
    
    def _get_next_mode(self, x: np.ndarray, t: float, current_mode: int) -> int:
        """
        Determine the next mode to switch to.
        
        Parameters
        ----------
        x : np.ndarray
            Current state.
        t : float
            Current time.
        current_mode : int
            Current active mode.
            
        Returns
        -------
        int
            Index of next mode.
        """
        if self.switching_map is not None:
            try:
                next_mode = self.switching_map(x, t, current_mode)
                if 0 <= next_mode < self.n_modes:
                    return next_mode
                else:
                    warnings.warn(f"switching_map returned invalid mode {next_mode}, cycling to next mode")
            except Exception as e:
                warnings.warn(f"switching_map evaluation failed: {e}, cycling to next mode")
        
        # Default: cycle to next mode
        return (current_mode + 1) % self.n_modes
    
    def _dynamics(self, t: float, x: np.ndarray, mode: int, control: Optional[Callable]) -> np.ndarray:
        """
        Compute state derivative dx/dt for current mode.
        
        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray
            Current state vector.
        mode : int
            Active mode index.
        control : callable or None
            Control input function u(t) -> np.ndarray.
            
        Returns
        -------
        np.ndarray
            State derivative dx/dt.
        """
        # Get control input
        if self.autonomous or control is None:
            u = np.zeros(self.n_inputs)
        else:
            u = control(t)
            u = np.asarray(u, dtype=self.dtype).flatten()
            if u.shape[0] != self.n_inputs:
                raise ValueError(f"Control input has {u.shape[0]} elements, expected {self.n_inputs}")
        
        # Compute linear part: A @ x + B @ u
        dx = self.A[mode] @ x
        if not self.autonomous:
            dx = dx + self.B[mode] @ u
        
        # Add nonlinear part
        dx = dx + self.f[mode](x, u, t)
        
        return dx
    
    def simulate_sequence(
        self,
        x0: Union[np.ndarray, torch.Tensor],
        mode_sequence: List[int],
        durations: Union[List[float], np.ndarray],
        controls: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        dt: Optional[float] = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        method: str = 'RK45'
    ) -> Dict:
        """
        Simulate system with a predefined sequence of modes, durations, and controls.
        
        This method is designed for executing a predetermined switching schedule with
        piecewise constant control inputs - ideal for evaluating neural network outputs
        or optimal control sequences.
        
        Parameters
        ----------
        x0 : np.ndarray or torch.Tensor
            Initial state vector, shape (n_states,).
        mode_sequence : list of int
            Sequence of mode indices to execute, shape (n_phases,).
        durations : list of float or np.ndarray
            Duration for each phase, shape (n_phases,). Must be positive.
        controls : list of np.ndarray or np.ndarray, optional
            Control input for each phase. Can be:
            - List of arrays, each shape (n_inputs,) - one per phase
            - 2D array of shape (n_phases, n_inputs)
            If None and system is not autonomous, zero control is applied.
        dt : float, optional
            Fixed time step for output. If None, adaptive stepping is used.
        rtol : float, default=1e-6
            Relative tolerance for ODE solver.
        atol : float, default=1e-9
            Absolute tolerance for ODE solver.
        method : str, default='RK45'
            ODE solver method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA').
            
        Returns
        -------
        dict
            Simulation results containing:
            - 't': np.ndarray, time points
            - 'x': np.ndarray, state trajectory (n_states, n_points)
            - 'mode': np.ndarray, active mode at each time point
            - 'u': np.ndarray, control input trajectory (n_inputs, n_points) if controls provided
            - 'phase_boundaries': list of float, cumulative time at phase transitions
            - 'success': bool, whether simulation completed successfully
            - 'message': str, status message
            
        Examples
        --------
        >>> # Simulate 3-phase sequence
        >>> mode_seq = [0, 1, 0]
        >>> durations = [1.0, 2.0, 1.5]
        >>> controls = [np.array([0.5]), np.array([0.2]), np.array([0.8])]
        >>> result = sim.simulate_sequence(x0, mode_seq, durations, controls)
        >>> print(f"Final state: {result['x'][:, -1]}")
        """
        # Convert initial state to numpy
        if TORCH_AVAILABLE and isinstance(x0, torch.Tensor):
            x0 = x0.detach().cpu().numpy()
        x0 = np.asarray(x0, dtype=np.float64).flatten()
        
        if x0.shape[0] != self.n_states:
            raise ValueError(f"Initial state has {x0.shape[0]} elements, expected {self.n_states}")
        
        # Validate inputs
        mode_sequence = list(mode_sequence)
        n_phases = len(mode_sequence)
        
        if n_phases == 0:
            raise ValueError("mode_sequence must have at least one element")
        
        # Validate mode indices
        for i, mode in enumerate(mode_sequence):
            if not (0 <= mode < self.n_modes):
                raise ValueError(f"mode_sequence[{i}] = {mode} is invalid. Must be in [0, {self.n_modes-1}]")
        
        # Convert and validate durations
        if TORCH_AVAILABLE and isinstance(durations, torch.Tensor):
            durations = durations.detach().cpu().numpy()
        durations = np.asarray(durations, dtype=np.float64).flatten()
        
        if durations.shape[0] != n_phases:
            raise ValueError(f"durations has {durations.shape[0]} elements, expected {n_phases}")
        
        if np.any(durations <= 0):
            raise ValueError("All durations must be positive")
        
        # Process controls
        if controls is None:
            if not self.autonomous:
                # Zero control
                controls_array = np.zeros((n_phases, self.n_inputs))
            else:
                controls_array = np.zeros((n_phases, 0))
        else:
            # Convert to array
            if TORCH_AVAILABLE and isinstance(controls, torch.Tensor):
                controls = controls.detach().cpu().numpy()
            
            if isinstance(controls, list):
                controls_array = np.array([np.asarray(c, dtype=np.float64).flatten() for c in controls])
            else:
                controls_array = np.asarray(controls, dtype=np.float64)
            
            # Validate shape
            if controls_array.ndim == 1:
                # Single control for all phases
                if controls_array.shape[0] != self.n_inputs:
                    raise ValueError(f"Control has {controls_array.shape[0]} elements, expected {self.n_inputs}")
                controls_array = np.tile(controls_array, (n_phases, 1))
            elif controls_array.ndim == 2:
                if controls_array.shape[0] != n_phases:
                    raise ValueError(f"controls has {controls_array.shape[0]} phases, expected {n_phases}")
                if controls_array.shape[1] != self.n_inputs:
                    raise ValueError(f"controls has {controls_array.shape[1]} inputs, expected {self.n_inputs}")
            else:
                raise ValueError(f"controls must be 1D or 2D, got shape {controls_array.shape}")
        
        # Reset history
        self.reset_history()
        
        # Compute phase boundaries (cumulative time)
        phase_boundaries = np.concatenate([[0], np.cumsum(durations)])
        t_total = phase_boundaries[-1]
        
        # Storage for results
        all_t = [0.0]
        all_x = [x0]
        all_modes = [mode_sequence[0]]
        all_u = []
        
        current_state = x0.copy()
        success = True
        message = "Simulation completed successfully"
        
        try:
            # Simulate each phase
            for phase_idx in range(n_phases):
                mode = mode_sequence[phase_idx]
                duration = durations[phase_idx]
                u_phase = controls_array[phase_idx]
                
                t_start = phase_boundaries[phase_idx]
                t_end = phase_boundaries[phase_idx + 1]
                
                # Create constant control function for this phase
                def control_func(t, u_const=u_phase):
                    return u_const
                
                # Define dynamics for this phase
                def phase_dynamics(t, x):
                    return self._dynamics(t, x, mode, control_func)
                
                # Time points for this phase
                if dt is not None:
                    t_eval_phase = np.arange(t_start, t_end + dt/2, dt)
                    # Ensure we include the end point
                    if t_eval_phase[-1] < t_end - atol:
                        t_eval_phase = np.append(t_eval_phase, t_end)
                else:
                    t_eval_phase = None
                
                # Integrate this phase
                sol = solve_ivp(
                    phase_dynamics,
                    (t_start, t_end),
                    current_state,
                    method=method,
                    t_eval=t_eval_phase,
                    rtol=rtol,
                    atol=atol,
                    dense_output=True
                )
                
                if not sol.success:
                    success = False
                    message = f"ODE solver failed at phase {phase_idx}: {sol.message}"
                    break
                
                # Store results (skip first point to avoid duplicates)
                all_t.extend(sol.t[1:])
                all_x.extend(sol.y.T[1:])
                all_modes.extend([mode] * (len(sol.t) - 1))
                
                # Store control inputs
                if not self.autonomous:
                    for _ in range(len(sol.t) - 1):
                        all_u.append(u_phase)
                
                # Update state for next phase
                current_state = sol.y[:, -1]
        
        except Exception as e:
            success = False
            message = f"Simulation error: {str(e)}"
            warnings.warn(message)
        
        # Convert to arrays
        t_result = np.array(all_t)
        x_result = np.array(all_x).T  # Shape: (n_states, n_points)
        mode_result = np.array(all_modes)
        
        result = {
            't': t_result,
            'x': x_result,
            'mode': mode_result,
            'phase_boundaries': phase_boundaries.tolist(),
            'n_phases': n_phases,
            'success': success,
            'message': message
        }
        
        if not self.autonomous and all_u:
            u_result = np.array(all_u).T  # Shape: (n_inputs, n_points)
            result['u'] = u_result
        
        # Store in history
        self.t_history = t_result
        self.x_history = x_result
        self.mode_history = mode_result
        self.switch_times = phase_boundaries.tolist()
        
        return result
    
    def simulate_mpc(
        self,
        x0: Union[np.ndarray, torch.Tensor],
        controller: SwiLinNN,
        dt_step: float,
        Nsim: Optional[int] = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        method: str = 'RK45',
        apply_first_only: bool = True,
        verbose: bool = False
    ) -> Dict:
        """
        Simulate system in Model Predictive Control (MPC) fashion.
        
        At each MPC iteration, the controller is called with the current state to compute
        the optimal control sequence and mode durations over a receding horizon. The first
        control action is then applied for one time step, and the process repeats.
        
        This enables closed-loop control where the optimization is resolved at each step
        based on the current state, mimicking real-time MPC operation.
        
        Parameters
        ----------
        x0 : np.ndarray or torch.Tensor
            Initial state vector, shape (n_states,).
        controller : SwiLinNN
            MPC controller function with signature:
                controller(x_current, t_current) -> dict
            Must return a dictionary containing:
                - 'mode_sequence': list of int, mode indices
                - 'durations': np.ndarray, time duration for each phase
                - 'controls': np.ndarray, control inputs for each phase (n_phases, n_inputs)
            The controller solves the optimal control problem over the horizon starting
            from x_current at t_current.
        dt_step : float
            Time step between MPC updates (sampling time). The first action from the
            controller is applied for this duration before reoptimizing.
        Nsim : int
            Number of simulation steps to perform. If None, continues indefinitely
            (use with caution - may run forever).
        rtol : float, default=1e-6
            Relative tolerance for ODE solver.
        atol : float, default=1e-9
            Absolute tolerance for ODE solver.
        method : str, default='RK45'
            ODE solver method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA').
        apply_first_only : bool, default=True
            If True, only the first control action is applied for dt_step.
            If False, the entire first phase from the controller is applied
            (using its duration, which may differ from dt_step).
        verbose : bool, default=False
            If True, prints progress information at each MPC iteration.
            
        Returns
        -------
        dict
            Simulation results containing:
            - 't': np.ndarray, time points
            - 'x': np.ndarray, state trajectory (n_states, n_points)
            - 'mode': np.ndarray, active mode at each time point
            - 'u': np.ndarray, control input trajectory (n_inputs, n_points) if not autonomous
            - 'mpc_update_times': list of float, times when controller was called
            - 'n_iterations': int, number of MPC iterations performed
            - 'success': bool, whether simulation completed successfully
            - 'message': str, status message
            
        Examples
        --------
        >>> # Define MPC controller (e.g., using neural network or optimization)
        >>> def my_controller(x, t):
        ...     # Solve optimal control problem
        ...     mode_seq, deltas, u = optimize_trajectory()
        ...     return {
        ...         'mode_sequence': mode_seq,
        ...         'durations': durations,
        ...         'controls': controls
        ...     }
        >>> 
        >>> # Run MPC simulation for 100 steps with 0.1s sampling time
        >>> result = sim.simulate_mpc(x0, controller=my_controller,
        ...                           dt_step=0.1, n_steps=100)
        >>> print(f"Final state: {result['x'][:, -1]}")
        
        Notes
        -----
        The controller is called at times: t=0, dt_step, 2*dt_step, ..., (Nsim-1)*dt_step.
        At each call, it receives the current state and must return a control plan over
        the horizon [t, t+t_horizon], where t_horizon is the prediction horizon length. 
        Only the first action is applied before reoptimizing.
        
        This implements a receding horizon control strategy commonly used in MPC.
        """
        # Convert initial state to numpy
        if TORCH_AVAILABLE and isinstance(x0, torch.Tensor):
            x0 = x0.detach().cpu().numpy()
        x0 = np.asarray(x0, dtype=np.float64).flatten()
        
        if x0.shape[0] != self.n_states:
            raise ValueError(f"Initial state has {x0.shape[0]} elements, expected {self.n_states}")
        
        if dt_step <= 0:
            raise ValueError("dt_step must be positive")
        
        if Nsim is not None and Nsim <= 0:
            raise ValueError("Nsim must be positive or None")
        
        self.controller = controller
        
        # Reset history
        self.reset_history()
        
        # Storage for results
        all_t = [0.0]
        all_x = [x0]
        all_modes = []
        all_u = []
        mpc_update_times = [0.0]
        
        current_state = x0.copy()
        current_time = 0.0
        iteration = 0
        success = True
        message = "MPC simulation completed successfully"
        # Track the currently active mode across MPC iterations
        # Initialize to the first mode in the default sequence
        current_mode = self.mode_seq[0] if len(self.mode_seq) > 0 else 0
        
        # try:
        while True:
            # Check termination
            if Nsim is not None and iteration >= Nsim:
                break
            
            if verbose:
                print(f"MPC iteration {iteration}: t={current_time:.4f}, x={current_state}")
            
            # Call controller to get optimal plan
            # try:
            #     u, deltas = self.compute_policy(current_state)
            # except Exception as e:
            #     success = False
            #     message = f"Controller failed at iteration {iteration}: {str(e)}"
            #     break
            u, deltas = self.compute_policy(current_state)
            
            # Convert to numpy arrays
            if TORCH_AVAILABLE and isinstance(deltas, torch.Tensor):
                deltas = deltas.detach().cpu().numpy()
            if TORCH_AVAILABLE and isinstance(u, torch.Tensor):
                u = u.detach().cpu().numpy()
            
            deltas = np.asarray(deltas, dtype=self.dtype).flatten()
            u = np.asarray(u, dtype=self.dtype)
            
            print(f"  Controller output: deltas={deltas}, max delta={np.max(deltas)} at index {np.argmax(deltas)}")
            input()
            
            # If controller returned a flat control vector, try to reshape into (n_steps, n_inputs)
            if u.ndim == 1:
                if self.n_inputs > 0 and (u.size % self.n_inputs) == 0:
                    u = u.reshape(-1, self.n_inputs)
                elif u.size == self.n_inputs:
                    u = u.reshape(1, -1)
                else:
                    raise ValueError(
                        f"Controller returned 1D control vector of length {u.size}, "
                        f"which is not divisible by n_inputs={self.n_inputs}."
                    )
            
            # Determine what to apply: first action only or first phase
            if apply_first_only:
                # Apply first control for dt_step
                apply_duration = dt_step
                # Use current active mode for application
                apply_mode = current_mode
                apply_control = u[0] if not self.autonomous else np.zeros(0)
            else:
                # Apply entire first phase
                apply_duration = deltas[0]
                apply_mode = current_mode
                apply_control = u[0] if not self.autonomous else np.zeros(0)
            
                # Define control function for this step
            def control_func(t, u_const=apply_control):
                return u_const
            
            # Define dynamics for this step
            def step_dynamics(t, x):
                return self._dynamics(t, x, apply_mode, control_func)
            
            # Time span for this step
            t_start = current_time
            t_end = current_time + apply_duration
            
            # Ensure mode at initial time is recorded so lengths align
            if len(all_modes) == 0:
                all_modes.append(apply_mode)

                # Integrate for one step
            t_eval_step = np.linspace(t_start, t_end, max(2, int(apply_duration / dt_step * 10) + 1))
            
            sol = solve_ivp(
                step_dynamics,
                (t_start, t_end),
                current_state,
                method=method,
                t_eval=t_eval_step,
                rtol=rtol,
                atol=atol
            )
            
            if not sol.success:
                success = False
                message = f"ODE solver failed at iteration {iteration}: {sol.message}"
                break
            
            # Store results (skip first point to avoid duplicates)
            all_t.extend(sol.t[1:])
            all_x.extend(sol.y.T[1:])
            all_modes.extend([apply_mode] * (len(sol.t) - 1))
            
            if not self.autonomous:
                for _ in range(len(sol.t) - 1):
                    all_u.append(apply_control)
            
            # Update state and time
            current_state = sol.y[:, -1]
            current_time = t_end
            iteration += 1
            mpc_update_times.append(current_time)
            # Update active mode based on controller-provided phase durations
            try:
                # deltas are durations for phases starting at t_start (plan origin)
                cumulative = np.cumsum(deltas)
                # Find which phase contains the time offset apply_duration
                phase_idx = int(np.searchsorted(cumulative, apply_duration, side='right'))
                if phase_idx < len(self.mode_seq):
                    next_mode = self.mode_seq[phase_idx]
                else:
                    next_mode = self.mode_seq[-1]
                current_mode = next_mode
            except Exception:
                # If deltas are malformed, keep current_mode unchanged
                pass
        
        # except Exception as e:
        #     success = False
        #     message = f"MPC simulation error: {str(e)}"
        #     warnings.warn(message)
        
        # Convert to arrays
        t_result = np.array(all_t)
        x_result = np.array(all_x).T  # Shape: (n_states, n_points)
        mode_result = np.array(all_modes)
        
        result = {
            't': t_result,
            'x': x_result,
            'mode': mode_result,
            'mpc_update_times': mpc_update_times,
            'n_iterations': iteration,
            'success': success,
            'message': message
        }
        
        if not self.autonomous and all_u:
            u_result = np.array(all_u).T  # Shape: (n_inputs, n_points)
            result['u'] = u_result
        
        # Store in history
        self.t_history = t_result
        self.x_history = x_result
        self.mode_history = mode_result
        self.switch_times = mpc_update_times
        
        return result
    
    def compute_policy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the optimal control inputs and phase durations for state x.
        
        Parameters
        ----------
        x : np.ndarray
            Current state (n_states,).
            
        Returns
        -------
        u_opt : np.ndarray
            Optimal control inputs (n_phases, n_inputs).
        delta_opt : np.ndarray
            Optimal phase durations (n_phases,).
        """
        # Prepare input tensor
        T  = self.controller.sys.time_horizon
        n_phases = self.controller.n_phases
        
        # start_deploy = time.time()
        # Forward pass through the network
        with torch.no_grad():
            # Convert state to tensor
            x_tensor = torch.tensor(x, dtype=self.controller.sys.dtype, device=self.controller.sys.device)  # shape (1, n_states)
            output = self.controller(x_tensor).flatten()
            output = output.cpu().numpy() # shape (n_phases * n_inputs + n_phases,)
        
        # Extract phase and control inputs
        n_control_outputs = n_phases * self.n_inputs
        pred_u = output[:n_control_outputs] # shape (n_phases * n_inputs,)
        pred_delta_raw = output[n_control_outputs:]
        
        # Apply softmax and scale deltas
        delta_normalized = softmax(pred_delta_raw, axis=-1)
        deltas = delta_normalized * T # shape (n_phases,)
        
        # Clip controls using tanh-based soft clipping to preserve gradients
        u_min = -1.0  # Define your lower bound
        u_max = 1.0   # Define your upper bound
        u_center = (u_max + u_min) / 2.0
        u_range = (u_max - u_min) / 2.0
        # Soft clipping: maps (-inf, inf) to (u_min, u_max) smoothly
        u = u_center + u_range * np.tanh(pred_u)
        
        # end_deploy = time.time()
        # print(f"Policy computation time: {end_deploy - start_deploy} seconds")
        
        return u, deltas
    
    def plot_trajectory(
        self,
        result: Optional[Dict] = None,
        state_indices: Optional[List[int]] = None,
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot state trajectory with mode switches highlighted.
        
        Parameters
        ----------
        result : dict, optional
            Simulation result from simulate(). If None, uses last simulation.
        state_indices : list of int, optional
            Which states to plot. If None, plots all states.
        figsize : tuple, default=(12, 8)
            Figure size.
        save_path : str, optional
            Path to save figure.
        show : bool, default=True
            Whether to display the figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if result is None:
            if len(self.t_history) == 0:
                raise ValueError("No simulation data available. Run simulate() first.")
            t = self.t_history
            x = self.x_history
            modes = self.mode_history
            # switch_times = self.switch_times
        else:
            t = result['t']
            x = result['x']
            modes = result['mode']
            # switch_times = result['switch_times']
        
        if state_indices is None:
            state_indices = list(range(self.n_states))
        
        n_plots = len(state_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, squeeze=False)
        axes = axes.flatten()
        
        # Plot each state
        for i, state_idx in enumerate(state_indices):
            ax = axes[i]
            
            # Plot state trajectory with different colors per mode
            for mode_idx in range(self.n_modes):
                mask = modes == mode_idx
                if np.any(mask):
                    ax.plot(t[mask], x[state_idx, mask], 'o', 
                           label=self.mode_names[mode_idx], markersize=2, linewidth=1.5)
            
            # Mark switch times
            # for switch_t in switch_times[1:]:  # Skip initial time
            #     ax.axvline(switch_t, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_ylabel(f'$x_{{{state_idx + 1}}}$', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        axes[-1].set_xlabel('Time $t$', fontsize=12)
        fig.suptitle('Switched System State Trajectory', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_phase_portrait(
        self,
        result: Optional[Dict] = None,
        state_pair: Tuple[int, int] = (0, 1),
        figsize: Tuple[float, float] = (8, 8),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot 2D phase portrait.
        
        Parameters
        ----------
        result : dict, optional
            Simulation result. If None, uses last simulation.
        state_pair : tuple of int, default=(0, 1)
            Indices of states to plot (x-axis, y-axis).
        figsize : tuple, default=(8, 8)
            Figure size.
        save_path : str, optional
            Path to save figure.
        show : bool, default=True
            Whether to display the figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if result is None:
            if len(self.t_history) == 0:
                raise ValueError("No simulation data available. Run simulate() first.")
            x = self.x_history
            modes = self.mode_history
        else:
            x = result['x']
            modes = result['mode']
        
        idx1, idx2 = state_pair
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectory colored by mode
        for mode_idx in range(self.n_modes):
            mask = modes == mode_idx
            if np.any(mask):
                ax.plot(x[idx1, mask], x[idx2, mask], 'o',
                       label=self.mode_names[mode_idx], markersize=3, linewidth=1.5)
        
        # Mark initial point
        ax.plot(x[idx1, 0], x[idx2, 0], 'go', markersize=10, label='Initial', zorder=5)
        
        # Mark final point
        ax.plot(x[idx1, -1], x[idx2, -1], 'rs', markersize=10, label='Final', zorder=5)
        
        ax.set_xlabel(f'$x_{{{idx1 + 1}}}$', fontsize=12)
        ax.set_ylabel(f'$x_{{{idx2 + 1}}}$', fontsize=12)
        ax.set_title('Phase Portrait', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


if __name__ == "__main__":
    # Example: Simple 2-mode switched system
    print("=" * 70)
    print("Switched System Simulator - Example")
    print("=" * 70)
    
    # Define two modes with different dynamics
    A1 = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    A2 = np.array([[-1.0, -0.5], [0.5, -1.0]])
    B1 = np.array([[1.0], [0.0]])
    B2 = np.array([[0.0], [1.0]])
    
    A = [A1, A2]
    B = [B1, B2]
    
    # Define switching condition: switch when ||x|| < 1.0
    def switching_condition(x, t, mode):
        norm = np.linalg.norm(x)
        # Switch from mode 0 when norm drops below 1.5
        if mode == 0:
            return norm < 1.5
        # Switch from mode 1 when norm drops below 0.8
        elif mode == 1:
            return norm < 0.8
        return False
    
    # Create simulator
    sim = SwitchedSystemSimulator(
        A=A,
        B=B,
        switching_conditions=[switching_condition, switching_condition],
        mode_names=['Oscillatory', 'Damped']
    )
    
    # Define control input (simple constant)
    def control(t):
        return np.array([0.1 * np.sin(t)])
    
    # Run simulation
    x0 = np.array([3.0, 0.5])
    result = sim.simulate(
        x0=x0,
        t_span=(0.0, 20.0),
        control=control,
        dt=0.05
    )
    
    print(f"\nSimulation completed: {result['message']}")
    print(f"Number of switches: {result['n_switches']}")
    print(f"Switch times: {[f'{t:.2f}' for t in result['switch_times']]}")
    print(f"Mode sequence: {result['switch_modes']}")
    
    # Plot results
    sim.plot_trajectory(result)
    sim.plot_phase_portrait(result)
