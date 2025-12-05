# Here the class SwiLin is defined
# It provides all the tools to instanciate a Switched Linear Optimization problem presented in the TAC paper.
# Starting from the switched linear system, it provides the cost function and its gradient w.r.t. the control input and the phases duration
# This is a PyTorch implementation converted from CasADi

from math import factorial
from numbers import Number

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

import scipy.io
from scipy.interpolate import Akima1DInterpolator
from scipy.linalg import block_diag


class SwiLin:
    def __init__(
        self,
        n_phases, 
        n_states, 
        n_inputs, 
        time_horizon, 
        auto=False, 
        propagation='exp', 
        plot="display", 
        device='cuda'
    ) -> None:
        """
        Set up the SwiLin class
        
        Args:
            n_phases        (int): Number of phases.
            n_states        (int): Number of states
            n_inputs        (int): Number of controls
            time_horizon    (float): Time horizon
            auto            (bool): Flag to set the optimization for autonomous systems
            propagation     (str): Propagation method ('exp' or 'int')
            plot            (str): Plot method ('display', 'save', or 'none')
            device          (str): Device to use ('cpu' or 'cuda')
        """
        # Check if number of phases is greater than 1
        if n_phases < 1:
            raise ValueError("The number of phases must be greater than 0.")
        self.n_phases = n_phases
        # Check if the number of states is greater than 0
        if n_states < 1:
            raise ValueError("The number of states must be greater than 0.")
        self.n_states = n_states
        # Check if the number of controls is greater than 0 or if the system is autonomous
        if n_inputs < 0:
            raise ValueError("The number of controls must be greater than 0.")
        self.n_inputs = n_inputs
        
        if time_horizon < 0:
            raise ValueError("The time horizon must be greater than 0.")
        self.time_horizon = time_horizon
        
        self.auto = auto
        
        # Define the propagation method: 'exp' for matrix exponential, 'int' for numerical integration
        if propagation not in ['exp', 'int']:
            raise ValueError("The propagation method must be 'exp' or 'int'.")
        self.propagation = propagation
        
        if plot not in ['display', 'save', 'none']:
            raise ValueError("The plot method must be display, save, or none.")
        self.plot = plot
        
        # Set device
        self.device = torch.device(device)
        
        # Default dtype (will be updated based on input data)
        self.dtype = torch.float64
        
        # Define the system's state variables as a tensor of shape (n_phases+1, n_states)
        self.x = torch.zeros(self.n_phases + 1, self.n_states, dtype=self.dtype, device=self.device) 
        # Control input defined as a tensor of shape (n_phases, n_inputs)
        if not self.auto:
            self.n_inputs = n_inputs
            self.phi_f = torch.zeros(self.n_phases, self.n_states, self.n_inputs, dtype=self.dtype, device=self.device)
            self.H = torch.zeros(self.n_phases, self.n_inputs, self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
            self.M = torch.zeros(self.n_phases, self.n_states, self.n_inputs, dtype=self.dtype, device=self.device)
            self.K = torch.zeros(self.n_phases, self.n_inputs, self.n_inputs, dtype=self.dtype, device=self.device)
        else:
            self.n_inputs = 0   
            
        # Initialize the matrices as 3D tensors of shapes (n_phases, whatever, whatever)
        self.E = torch.zeros(self.n_phases, self.n_states, self.n_states, dtype=self.dtype, device=self.device)
        self.L = torch.zeros(self.n_phases, self.n_states, self.n_states, dtype=self.dtype, device=self.device)
        self.S = torch.zeros(self.n_phases + 1, self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
        self.Sr = []
        # self.C = []
        # self.N = []
        # self.D = []
        # self.G = []
        self.T = torch.zeros(self.n_phases, self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
        self.Sr_int = []
    
    def load_model(self, model) -> None:
        """
        Load the switched linear model 
        
        Args:
            model   (dict): Dictionary that stores all the informations about the switched linear model
        """
        A = []
        B = []
        for i in range(self.n_phases):
            id = i % len(model['A'])
            # Convert numpy arrays to torch tensors
            A.append(torch.tensor(model['A'][id], dtype=self.dtype, device=self.device))
            # Check if the input matrix is empty
            if model['B']:
                B.append(torch.tensor(model['B'][id], dtype=self.dtype, device=self.device))
            else:
                B.append(torch.zeros((model['A'][id].shape[0], 1), dtype=self.dtype, device=self.device))
              
        self.A = A
        self.B = B
        
        # Define the number of modes of the system
        self.n_modes = len(A)
    
    def load_weights(self, Q, R, E) -> None:
        """
        Load the weight matrices
        
        Args:
            Q   (torch.Tensor or np.ndarray): The weight matrix for the state.
            R   (torch.Tensor or np.ndarray): The weight matrix for the control.
            E   (torch.Tensor or np.ndarray): The weight matrix for the terminal state.
        """
        
        # Convert numpy arrays to torch tensors
        # Convert Q to tensor if it's numpy and augment by 1 (add zero row/col)
        if isinstance(Q, np.ndarray):
            self.Q = torch.tensor(Q, dtype=self.dtype, device=self.device)
        else:
            self.Q = Q.to(dtype=self.dtype, device=self.device)
        
        if isinstance(R, np.ndarray):
            self.R = torch.tensor(R, dtype=self.dtype, device=self.device)
        else:
            self.R = R.to(dtype=self.dtype, device=self.device)
        
        if isinstance(E, np.ndarray):
            self.E_term = torch.tensor(E, dtype=self.dtype, device=self.device)
        else:
            self.E_term = E.to(dtype=self.dtype, device=self.device)

    def integrator(self, func, t0, tf, *args):
        """
        Integrates f(t) between t0 and tf using the given function func using the composite Simpson's 1/3 rule.
        
        Args:
            func    (callable): The function that describes the system dynamics
            t0      (torch.Tensor): The initial time
            tf      (torch.Tensor): The final time
            *args   (torch.Tensor): Additional arguments to pass to the function
            
        Returns:
            integral    (torch.Tensor): The result of the integration
        """
        # Number of steps for the integration
        steps = 10
        
        # Check if args is not empty and set the input accordingly
        input = args[0] if args else None

        # Integration using the composite Simpson's 1/3 rule
        h = (tf - t0) / steps
        t = t0

        # Determine if the system is autonomous or not
        is_autonomous = input == 'auto'

        # Integration for autonomous systems
        if is_autonomous:
            S = func(t) + func(tf)
        else:
            # Determine if the input is a tensor
            is_tensor = input is not None and torch.is_tensor(input)
            # Integration for non-autonomous systems or general integrator
            S = func(t, input) + func(tf, input) if is_tensor else func(tf, t) + func(tf, tf)
    
        for k in range(1, steps):
            coefficient = 2 if k % 2 == 0 else 4
            t = t + h
            if is_autonomous:
                S = S + func(t) * coefficient
            else:
                # Determine if the input is a tensor
                is_tensor = input is not None and torch.is_tensor(input)
                # Integration for non-autonomous systems or general integrator
                S = S + func(t, input) * coefficient if is_tensor else func(tf, t)*coefficient

        integral = S * (h / 3)
        
        return integral
    
    def compute_integral(self, A, B, tmin, tmax):
        """
        Computes the forced evolution of the system's state using PyTorch for operations.
        
        Args:
        A (torch.Tensor): The system matrix.
        B (torch.Tensor): The input matrix.
        tmin (torch.Tensor): The start time for the integration.
        tmax (torch.Tensor): The end time for the integration.
        
        Returns:
        torch.Tensor: The result of the integral computation.
        """
        
        # Define the function to be integrated
        def f(s):
            return self.expm(A, (tmax - s)) @ B
    
        integral_result = self.integrator(f, tmin, tmax, 'auto')
        
        return integral_result
    
    def expm(self, A, delta):
        """
        Computes the matrix exponential of A * delta using PyTorch Taylor series.
        
        Args:
            A (torch.Tensor): "A" matrix the mode for which to compute the matrix exponential.
            delta (torch.Tensor or float): time variable
            
        Returns:
            exp_max (torch.Tensor): The computed matrix exponential.
        """        
        
        n = A.shape[0]  # Size of matrix A
                
        # Number of terms for the Taylor series expansion
        # num_terms = self._get_n_terms_expm_approximation()
        
        # Convert A to tensor if it's numpy
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=self.dtype, device=self.device)
        
        # Ensure delta is a tensor
        if not torch.is_tensor(delta):
            delta = torch.tensor(delta, dtype=self.dtype, device=self.device)
        
        # Compute the matrix exponential using the built-in PyTorch function
        result = torch.linalg.matrix_exp(A * delta)
    
        return result
    
    def _get_n_terms_expm_approximation(self):
        threshold = 1e-3
        
        n_terms_min = 6
        n_terms_max = 100
        
        err = self.time_horizon**n_terms_min / factorial(n_terms_min)
        
        for n_terms in range(n_terms_min+1, n_terms_max):
            if err < threshold:
                return n_terms_min
            
            err *= self.time_horizon / n_terms
            
        return n_terms_max
    
    def _mat_exp_prop_exp(self, index, u_i, delta_i):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.
        u_i     (torch.Tensor): The control input for the i-th mode.
        delta_i (torch.Tensor): The phase duration for the i-th mode.

        Returns:
        Ei      (torch.Tensor): The matrix exponential of Ai*delta_i.
        phi_f_i (torch.Tensor): The integral part multiplied by the control input ui.
        Hi      (list): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        Li      (torch.Tensor): Matrix for the cost function
        Mi      (torch.Tensor): Matrix for the cost function
        Ri      (torch.Tensor): Matrix for the cost function
        
        """        
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Create a big matrix of dimensions (3n+m)x(3n+m) 
        C = torch.zeros((3*self.n_states + self.n_inputs, 3*self.n_states + self.n_inputs), 
                        dtype=self.dtype, device=self.device)
        # Fill the matrix
        C[:self.n_states, :self.n_states] = -A.T
        C[:self.n_states, self.n_states:2*self.n_states] = torch.eye(self.n_states, dtype=self.dtype, device=self.device)
        C[self.n_states:2*self.n_states, self.n_states:2*self.n_states] = -A.T
        C[self.n_states:2*self.n_states, 2*self.n_states:3*self.n_states] = self.Q
        C[2*self.n_states:3*self.n_states, 2*self.n_states:3*self.n_states] = A
        if not self.auto:
            C[2*self.n_states:3*self.n_states, 3*self.n_states:] = B
        
        # Compute matrix exponential
        exp_C = self.expm(C, delta_i)
        
        # Extract the instrumental matrices from the matrix exponential
        F3 = exp_C[2*self.n_states:3*self.n_states, 2*self.n_states:3*self.n_states]
        G2 = exp_C[self.n_states:2*self.n_states, 2*self.n_states:3*self.n_states]
        Ei = F3
        Li = F3.T @ G2        
        
        # Distinct case for autonomous systems
        # Extract the control input
        if not self.auto:
            G3 = exp_C[2*self.n_states:3*self.n_states, 3*self.n_states:]
            H2 = exp_C[self.n_states:2*self.n_states, 3*self.n_states:]
            K1 = exp_C[:self.n_states, 3*self.n_states:]
            
            phi_f_i_ = G3
            
            ui_col = u_i.reshape(-1, 1) if u_i.dim() == 1 else u_i
            phi_f_i = phi_f_i_ @ ui_col
            
            Mi = F3.T @ H2
            
            Ri = (B.T @ F3.T @ K1) + (B.T @ F3.T @ K1).T
            
            # Create the H matrix related to the i-th mode (only for the non-autonomous case)
            Hi = []
            
            # Fill the Hk matrix with the k-th column of phi_f_i_ (integral term)
            Hi = torch.zeros(self.n_inputs, self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
            for k in range(self.n_inputs):
                Hi[k, :self.n_states, self.n_states] = phi_f_i_[:, k]
        
            return Ei, phi_f_i, Hi, Li, Mi, Ri
        else:
            return Ei, Li
        
    def _mat_exp_prop_int(self, index):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.

        Returns:
        Ei      (torch.Tensor): The matrix exponential of Ai*delta_i.
        phi_f_i (torch.Tensor): The integral part multiplied by the control input ui.
        Hi      (list): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        """        
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
                
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Compute matrix exponential
        Ei = self.expm(A, delta_i)
        
        # Distinct case for autonomous systems
        # Extract the control input
        if self.n_inputs > 0:
            ui = self.u[index]
        
            # Compute the integral of the system matrix and input matrix over the time interval
            phi_f_i_ = self.compute_integral(A, B, 0, delta_i)
            
            phi_f_i = phi_f_i_ @ ui
        
        # Create the H matrix related to the i-th mode (only for the non-autonomous case)
        if self.n_inputs > 0:
            Hi = []
            
            # Fill the Hk matrix with the k-th column of phi_f_i_ (integral term)
            for k in range(ui.shape[0]):
                Hk = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
                Hk[:self.n_states, self.n_states] = phi_f_i_[:, k]
                Hi.append(Hk)
        
            return Ei, phi_f_i, Hi, torch.zeros(0, dtype=self.dtype, device=self.device), torch.zeros(0, dtype=self.dtype, device=self.device), torch.zeros(0, dtype=self.dtype, device=self.device)
        else:
            return Ei, torch.zeros(0, device=self.device), 0, torch.zeros(0, device=self.device), torch.zeros(0, device=self.device), torch.zeros(0, device=self.device)
        
    def mat_exp_prop(self, index, u_i, delta_i):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.
        Q       (torch.Tensor): Weight matrix
        R       (torch.Tensor): Weight matrix

        Returns:
        Ei      (torch.Tensor): The matrix exponential of Ai*delta_i.
        phi_f_i (torch.Tensor): The integral part multiplied by the control input ui.
        Hi      (list): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        Li      (torch.Tensor): Matrix for the cost function
        Mi      (torch.Tensor): Matrix for the cost function
        Ri      (torch.Tensor): Matrix for the cost function
        
        """        
        if self.propagation == 'exp':
            return self._mat_exp_prop_exp(index, u_i, delta_i)
        else:
            return self._mat_exp_prop_int(index)
        
    def _propagate_state(self, x0):
        """
        Forward propagate the states through all phases.
        """
        # Set initial state
        self.x[0] = x0
        for i in range(self.n_phases):
            if not self.auto:
                self.x[i+1] = self.E[i] @ self.x[i] + self.phi_f[i].squeeze(-1)
            else:
                self.x[i+1] = self.E[i] @ self.x[i]
        
    def transition_matrix(self, phi_a, phi_f):
        """
        Computes the transition matrix for the given index.
        
        Args:
        phi_a (torch.Tensor): The matrix exponential of the system matrix.
        phi_f (torch.Tensor): The integral term.
        
        Returns:
        phi (torch.Tensor): The transition matrix.
        
        """
        phi = torch.zeros(self.n_states+1, self.n_states+1, dtype=self.dtype, device=self.device)
        
        # Distinct case for autonomous and non-autonomous systems
        if not self.auto:
            phi[:self.n_states, :self.n_states] = phi_a
            phi[:self.n_states, self.n_states:self.n_states+1] = phi_f
            phi[-1, -1] = 1
        else:
            phi[:self.n_states, :self.n_states] = phi_a
            phi[-1, -1] = 1
        
        return phi
           
    def D_matrix(self, index, Q):
        """
        Computes the D matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (torch.Tensor): The weight matrix.
        
        Returns:
        D (list): The D matrix.
        
        """
        # Define the system matrices for the given index
        B = self.B[index]
        A = self.A[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Create the D matrix related to the i-th mode
        D = []
        
        # Fill the D matrix with the Dij terms
        for k in range(ui.shape[0]):
            def f(eta, u_input):
                phi_a_t = self.expm(A, eta)
                phi_f_t = self.compute_integral(A, B, 0, eta)
                phi_t = self.transition_matrix(phi_a_t, phi_f_t @ u_input)
                
                Hij_t = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float32, device=self.device)
                Hij_t[:self.n_states, self.n_states] = phi_f_t[:, k]
            
                arg = (Hij_t.T @ Q @ phi_t) + (phi_t.T @ Q @ Hij_t)
                return 0.5 * arg
            
            Dij = self.integrator(f, 0, delta_i, ui)
            D.append(Dij)
        
        return D
    
    def _S_matrix_exp(self, index, u_i, delta_i, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):          The index of the mode.
        u_i     (torch.Tensor): The control input for the mode.
        delta_i (torch.Tensor): The phase duration for the mode.
        xr      (torch.Tensor): The reference state.
        
        Returns:
        S       (torch.Tensor): The S matrix.
        Optional:
        Sr      (torch.Tensor): The Sr matrix.
        
        """
        # Extract the autonomous and non-autonomous parts of the state
        phi_a = self.E[index]
        if not self.auto:
            phi_f = self.phi_f[index]
        else:
            phi_f = None
                
        # Extract the matrices for the integral term
        Li = self.L[index]
        if not self.auto:
            Mi = self.M[index]
            Ki = self.K[index]
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index - 1]
        
        phi_i = self.transition_matrix(phi_a, phi_f)
        
        S_int = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=self.dtype, device=self.device)
        S_int[:self.n_states, :self.n_states] = Li
        if not self.auto:
            ui_col = u_i.reshape(-1, 1) if u_i.dim() == 1 else u_i
            Mi_ui = Mi @ ui_col
            S_int[:self.n_states, self.n_states:] = Mi_ui
            S_int[self.n_states:, :self.n_states] = Mi_ui.T
            S_int[self.n_states:, self.n_states:] = ui_col.T @ Ki @ ui_col
        
        # If a reference state is given, compute both the Sr matrix and the S matrix
        if xr is not None:
            S = S_int + (phi_i.T @ S_prev @ phi_i)
            return S
        
        # Compute S matrix
        S = 0.5 * S_int + (phi_i.T @ S_prev @ phi_i)
        
        return S
    
    def _S_matrix_int(self, index, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):          The index of the mode.
        Q       (torch.Tensor): The weight matrix.
        xr      (torch.Tensor): The reference state.
        
        Returns:
        S       (torch.Tensor): The S matrix.
        Optional:
        Sr      (torch.Tensor): The Sr matrix.
        
        """
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Extract the autonomous and non-autonomous parts of the state
        phi_a = self.E[index]
        phi_f = self.phi_f[index]
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[0]
        
        if xr is not None:
            Sr_prev = self.Sr[0]
        
        # Compute the integral term
        def f(eta, u_input=None):
            phi_a_t = self.expm(A, eta)
            phi_f_t = self.compute_integral(A, B, 0, eta)
            
            if self.n_inputs == 0:
                phi_t = self.transition_matrix(phi_a_t, phi_f_t)
            elif self.n_inputs > 0:
                phi_t = self.transition_matrix(phi_a_t, phi_f_t @ u_input)
            else:
                raise ValueError("The number of controls must be greater than 0.")
            
            return phi_t.T @ self.Q @ phi_t
        
        if xr is not None:
            # Integral term that updates the Sr matrix
            def fr(eta, u_input=None):
                phi_a_t = self.expm(A, eta)
                phi_f_t = self.compute_integral(A, B, 0, eta)
                
                if self.n_inputs == 0:
                    phi_t = self.transition_matrix(phi_a_t, phi_f_t)
                elif self.n_inputs > 0:
                    phi_t = self.transition_matrix(phi_a_t, phi_f_t @ u_input)
                else:
                    raise ValueError("The number of controls must be greater than 0.")
                
                return phi_t.T @ Q @ xr
        
        # Compute the integral of the S matrix
        if self.n_inputs == 0:
            S_int = self.integrator(f, 0, delta_i, 'auto')
            # if a reference state is given, compute the Sr matrix
            if xr is not None:
                Sr_int = self.integrator(fr, 0, delta_i, 'auto')
        else:
            S_int = self.integrator(f, 0, delta_i, ui)
            # if a reference state is given, compute the Sr matrix
            if xr is not None:
                Sr_int = self.integrator(fr, 0, delta_i, ui)
        
        phi_i = self.transition_matrix(phi_a, phi_f)
        
        # If a reference state is given, compute both the Sr matrix and the S matrix
        if xr is not None:
            Sr = Sr_int + (phi_i.T @ Sr_prev)
            S = 0.5 * S_int + (phi_i.T @ S_prev @ phi_i)
            return S, Sr
        
        # Compute S matrix
        S = 0.5 * S_int + (phi_i.T @ S_prev @ phi_i)
        
        return S    
    
    def S_matrix(self, index, u_i, delta_i, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):          The index of the mode.
        u_i     (torch.Tensor): The control input for the mode.
        delta_i (torch.Tensor): The phase duration for the mode.
        xr      (torch.Tensor): The reference state.
        
        Returns:
        S       (torch.Tensor): The S matrix.
        Optional:
        Sr      (torch.Tensor): The Sr matrix.
        
        """
        if self.propagation == 'exp':
            return self._S_matrix_exp(index, u_i, delta_i, xr)
        else:
            return self._S_matrix_int(index, u_i, delta_i, xr)
        
    def C_matrix(self, index, Q):
        """
        Computes the C matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (torch.Tensor): The weight matrix.
        
        Returns:
        C (torch.Tensor): The C matrix.
        
        """
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Define the M matrix
        M = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float32, device=self.device)
        
        M[:self.n_states, :self.n_states] = A
        M[:self.n_states, self.n_states] = B @ ui
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index+1]
        
        C = 0.5*Q + M.T @ S_prev + S_prev @ M
        
        return C
           
    def N_matrix(self, index):
        """
        Computes the N matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        
        Returns:
        N (list): The N matrix.
        
        """
        # Initialize the N matrix of the current iteration
        N = []
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index+1]
        
        # Extract the H matrix of the current iteration
        H = self.H[index]
        
        for j in range(self.n_inputs):
            Hij = H[j]
            # Compute N matrix
            Nij = Hij.T @ S_prev + S_prev @ Hij
            N.append(Nij)
        
        return N
        
    def compute_G(self, u, delta):
        """
        Computes the G matrix.
        
        Args:
        u       (torch.Tensor): The control inputs.
        delta   (torch.Tensor): The phase durations.
        
        Returns:
        G (torch.Tensor): The G matrix.
        
        """
        
        G = 0
        # Fast vectorized computation of G = 0.5 * sum_i delta[i] * u[i]^T R u[i]
        if self.n_inputs == 0:
            return G

        # Convert u to tensor of shape (n_phases, n_inputs)
        if not torch.is_tensor(u):
            u = torch.stack([
            torch.tensor(ui, dtype=self.dtype, device=self.device) if not torch.is_tensor(ui)
            else ui.to(dtype=self.dtype, device=self.device)
            for ui in u
            ])
        else:
            u = u.to(dtype=self.dtype, device=self.device)

        # Convert delta to 1D tensor (n_phases,)
        if not torch.is_tensor(delta):
            delta = torch.tensor(delta, dtype=self.dtype, device=self.device)
        else:
            delta = delta.to(dtype=self.dtype, device=self.device)
        delta = delta.view(-1)

        # Ensure R has correct dtype/device
        R = self.R.to(dtype=self.dtype, device=self.device)

        # uR shape: (n_phases, n_inputs); elementwise multiply with u and sum over inputs gives u^T R u per phase
        per_phase = (u @ R) * u
        per_phase = per_phase.sum(dim=1)  # shape (n_phases,)

        # Accumulate weighted by durations
        G = 0.5 * (per_phase * delta).sum()
            
        return G
        
    def cost_function(
        self, 
        u_all = None, 
        delta_all = None, 
        x0 = None
    ):
        """
        Compute total cost J.
        
        J = G0 + [x0; 1]' * S_0 * [x0; 1]
        
        Args:
        u_all       (list of torch.Tensor): The control inputs for all phases.
        delta_all   (list of torch.Tensor): The phase durations for all phases.
        x0          (torch.Tensor): The initial state.
        
        Returns:
        callable: A function that takes (*u_list, *delta_list, x0) and returns the cost J.
        """
       
        # Ensure x0 is a tensor
        if not isinstance(x0, torch.Tensor):
            x0_tensor = torch.tensor(x0, dtype=self.dtype, device=self.device)
        else:
            x0_tensor = x0
            
        # Check if auto is True and set the inputs to zero
        if self.auto and u_all is None:
            u_all = torch.zeros((self.n_phases, 1), dtype=self.dtype, device=self.device)
        
        # Forward propagation
        self.forward_propagate(u_all, delta_all)
        
        # Backward propagation
        self.backward_propagate(u_all, delta_all, x0_tensor)
        
        # Control cost
        if self.auto:
            G0 = 0.0
        else:
            G0 = self.compute_G(u_all, delta_all)
        
        # Initial state augmentation
        x0_aug = torch.cat([x0_tensor, torch.ones(1, device=self.device, dtype=self.dtype)])

        J = 0.5 * x0_aug @ self.S[0] @ x0_aug + G0
            
        return J
        
    def grad_cost_function(self, index, R):
        """
        Computes the gradient of the cost function.
        
        Args:
        index (int): The index of the mode.
        R (torch.Tensor): The weight matrix.
        
        Returns:
        du (list): The gradient of the cost function with respect to the control input.
        d_delta (torch.Tensor): The gradient of the cost function with respect to the phase duration.
        
        """
        
        # Create the augmented state vectors
        x_aug = torch.zeros(self.n_states + 1, dtype=torch.float32, device=self.device)
        x_next_aug = torch.zeros(self.n_states + 1, dtype=torch.float32, device=self.device)
        x_aug[:self.n_states] = self.x[index]
        x_aug[self.n_states] = 1
        x_next_aug[:self.n_states] = self.x[index+1]
        x_next_aug[self.n_states] = 1

        # Extract the control input
        ui = self.u[index]

        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Extract the C matrix of the current iteration
        C = self.C[index]
        
        # Extract the N matrix of the current iteration
        N = self.N[index]
        
        # Extract the D matrix of the current iteration
        D = self.D[index]
        
        # Compute the gradient of the cost function with respect to the control input
        du = []
        for j in range(self.n_inputs):
            # D and N are lists (one matrix per input). Index the j-th element.
            Dij = D[j] if isinstance(D, (list, tuple)) else D
            Nij = N[j] if isinstance(N, (list, tuple)) else N

            term_r = ui[j] * R[j, j] * delta_i
            term_d = x_aug.T @ Dij @ x_aug if Dij is not None else 0
            term_n = x_next_aug.T @ Nij @ x_next_aug if Nij is not None else 0

            du_j = term_r + term_d + term_n
            du.append(du_j)

        # Compute the gradient of the cost function with respect to the phase duration
        d_delta = 0.5 * (ui.T @ R @ ui) + (x_next_aug.T @ C @ x_next_aug)
        
        return du, d_delta
    
    def forward_propagate(
        self, 
        u_all: torch.Tensor,
        delta_all: torch.Tensor,
        xr=None
    ) -> None:
        """
        Forward propagate the states through all phases.
        
        Args:
        u_all (torch.Tensor): Control inputs for all phases.
        delta_all (torch.Tensor): Phase durations for all phases.
        xr (torch.Tensor, optional): Reference state.
        """       
        
        for i in range(self.n_phases):
            u_i = u_all[i]
            delta_i = delta_all[i]
            # Compute the matrix exponential properties
            if self.auto:
                Ei, Li = self.mat_exp_prop(i, u_i, delta_i)
            else:
                Ei, phi_f_i, Hi, Li, Mi, Ki = self.mat_exp_prop(i, u_i, delta_i)
            self.E[i] = Ei
            self.L[i] = Li
            if not self.auto:
                self.phi_f[i] = phi_f_i
                self.H[i] = Hi

                self.M[i] = Mi
                self.K[i] = Ki
                
    def backward_propagate(
        self,
        u_all: torch.Tensor,
        delta_all: torch.Tensor,
        x0: torch.Tensor,
        xr=None
    ) -> None:
        
        # Initialize the S matrix with the terminal cost (if needed)
        # Augment E_term by adding a zero row and column
        E_ = torch.zeros((self.n_states + 1, self.n_states + 1), dtype=self.dtype, device=self.device)
        E_[:self.n_states, :self.n_states] = self.E_term
        self.S[-1] = 0.5*E_
        if xr is not None:
            if isinstance(xr, np.ndarray):
                xr = torch.tensor(xr, dtype=self.dtype, device=self.device)
            xr_aug = torch.cat([xr, torch.ones(1, dtype=self.dtype, device=self.device)])
            self.Sr.append(0.5*self.E_term @ xr_aug)

        for i in range(self.n_phases-1, -1, -1):
            u_i = u_all[i]
            delta_i = delta_all[i]
            
            # Compute the S matrix (and Sr if needed)
            if xr is not None:
                # Compute the S and Sr matrices
                S, Sr = self.S_matrix(i, u_i, delta_i, xr)
                self.S[i] = S
                self.Sr[i] = Sr
            else:
                # Compute the S matrix
                S = self.S_matrix(i, u_i, delta_i)
                self.S[i] = S
        
        # Compute the C and N matrices
        # for i in range(self.n_phases):
        #     C = self.C_matrix(i, Q_)
        #     self.C.append(C)
        #     if self.n_inputs > 0:
        #         N = self.N_matrix(i)
        #         self.N.append(N)
                
        # Propagate the state using the computed matrices.
        self._propagate_state(x0)
       
    def state_extraction(self, delta_opt, *args):
        """
        Extract the optimal values of the state trajectory based on the optimized values of u and delta
        """    
        
        # Check if args is not empty and set the input accordingly
        u_opt = args[0] if args else None   
        
        x_opt = []
        
        # Set delta values
        for i in range(self.n_phases):
            self.delta[i] = delta_opt[i] if torch.is_tensor(delta_opt[i]) else torch.tensor(delta_opt[i], device=self.device)
        
        # Set u values if non-autonomous
        if self.n_inputs > 0 and u_opt is not None:
            if isinstance(u_opt[0], Number):
                u_opt = np.reshape(u_opt, (-1, self.n_inputs))
            for i in range(self.n_phases):
                self.u[i] = torch.tensor(u_opt[i], dtype=torch.float32, device=self.device) if not torch.is_tensor(u_opt[i]) else u_opt[i]
        
        # Extract states
        for i in range(self.n_phases+1):
            x_opt.append(self.x[i].detach().cpu().numpy() if torch.is_tensor(self.x[i]) else self.x[i])
                
        return x_opt
    
    def split_list_to_arrays(self, input_list, chunk_size):
        '''
        This function splits a list into chunks of the specified size.
        '''
        # Ensure the input list can be evenly divided by chunk_size
        if len(input_list) % chunk_size != 0:
            raise ValueError("The length of the input list must be divisible by the chunk size.")
        
        # Split the list into chunks of the specified size
        list_of_arrays = [np.array(input_list[i:i + chunk_size]) for i in range(0, len(input_list), chunk_size)]
        
        return list_of_arrays
    
    def plot_optimal_solution(self, delta_opt, *args, save=False, filename=None):
        """
        Plot the optimal state trajectory based on the optimized values of u and delta
        Plot the optimal control input if the system is non-autonomous
        """    
        # Check if the filename is provided
        if save and filename is None:
            raise ValueError("Filename must be provided for saving the plot.")
        
        # Check if args is not empty and set the input accordingly
        u_opt = args[0] if args else None   
        
        # Check if the state vector is part of the optimization
        if len(args) > 1:
            x_opt = args[1]
        else:
            x_opt = None
        
        if x_opt is None:
            # Extract the optimal state trajectory
            if self.n_inputs == 0:
                x_opt = self.state_extraction(delta_opt)
            else:
                x_opt = self.state_extraction(delta_opt, u_opt)
            
            x_opt_num = np.array([x_opt[i] if isinstance(x_opt[i], np.ndarray) else x_opt[i] for i in range(len(x_opt))])
            x_opt_num = x_opt_num.reshape(len(x_opt), -1)
        else:
            x_opt_ = self.split_list_to_arrays(x_opt, self.n_states)
            x_opt_num = np.vstack(x_opt_)

        if save:
            # Save data to a .mat file
            data_to_save = {
                'n_states': self.n_states,
                'time_horizon': self.time_horizon,
                'n_phases': self.n_phases,
                'trajectory': x_opt_num,
                'controls': u_opt,
                'phases_duration': delta_opt
            }

            scipy.io.savemat(filename + '.mat', data_to_save)
        
        # Plot the state trajectory
        # Create the time grid mesh
        tgrid = []
        points = 1000
        time = 0
        next_time = 0
        for i in range(self.n_phases):
            next_time = next_time + delta_opt[i]
            tgrid = np.concatenate((tgrid, np.linspace(time, next_time, points, endpoint=False)))
            time = time + delta_opt[i]
        tgrid = np.concatenate((tgrid, [self.time_horizon]))
        
        
        traj = np.zeros((len(tgrid), self.n_states))
        # If the state vector is part of the optimization, resample the state trajectory
        for i in range(self.n_states):
            interp = Akima1DInterpolator(
                np.linspace(0, self.time_horizon, x_opt_num.shape[0]),
                x_opt_num[:, i],
                method='akima',
                )
            tmp = interp(np.linspace(0, self.time_horizon, len(tgrid)))
            traj[:, i] = tmp
        
        fig, ax= plt.subplots()
        for i in range(self.n_states):  
            ax.plot(tgrid, traj[:, i], label=f'x{i+1}')  
        ax.set_xlim([0, self.time_horizon])
        # Add a legend
        ax.legend(loc='upper right')
        # Add vertical lines to identify phase changes instants
        time = 0
        for i in range(self.n_phases):
            time = time + delta_opt[i]
            plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)
        ax.set(xlabel='Time', ylabel='State')
        if self.plot == 'save':
            plt.savefig(filename + '_optimal_state.pdf', format='pdf', bbox_inches='tight')
        
        # Plot the control input if the system is non-autonomous
        
        if self.n_inputs > 0:
            fig, ax = plt.subplots(self.n_inputs, 1)
            u_opt_list = np.reshape(u_opt, (-1, self.n_inputs)).tolist()
            u_opt_list += u_opt_list[-1:]
            if self.n_inputs >= 2:
                for i in range(self.n_inputs):
                    # Extract the optimal control input at different time instants
                    input = [sublist[i] for sublist in u_opt_list]
                    ax[i].step(tgrid[::points], np.array(input), where='post', linewidth=2)
                    ax[i].set(xlabel='Time', ylabel='Input_'+str(i))
                    ax[i].set_xlim([0, self.time_horizon])
                    # Add vertical lines to identify phase changes instants
                    time = 0
                    for j in range(self.n_phases):
                        time = time + delta_opt[j]
                        ax[i].axvline(x=time, color='k', linestyle='--', linewidth=0.5)
            else:
                ax.step(tgrid[::points], np.array(u_opt_list), where='post', linewidth=2)
                ax.set(xlabel='Time', ylabel='Input')
                ax.set_xlim([0, self.time_horizon])
                # Add vertical lines to identify phase changes instants
                time = 0
                for i in range(self.n_phases):
                    time = time + delta_opt[i]
                    plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)   
        
        if self.plot == 'save':
            plt.savefig(filename + '_optimal_input.pdf', format='pdf', bbox_inches='tight')
        elif self.plot == 'display':
            plt.show()
