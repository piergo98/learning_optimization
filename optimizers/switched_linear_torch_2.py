# Here the class SwiLin is defined
# It provides all the tools to instanciate a Switched Linear Optimization problem presented in the TAC paper.
# Starting from the switched linear system, it provides the cost function and its gradient w.r.t. the control input and the phases duration

from math import factorial
from numbers import Number

import torch
import matplotlib.pyplot as plt
import numpy as np

import scipy.io
from scipy.interpolate import Akima1DInterpolator
from scipy.linalg import block_diag


class SwiLin:
    def __init__(self, n_phases, n_states, n_inputs, time_horizon, auto=False, propagation='exp', plot="display") -> None:
        """
        Set up the SwiLin class
        
        Args:
            n_phases          (int): Number of phases.
            n_states          (int): Number of states
            n_inputs          (int): Number of controls
            time_horizon    (float): Time horizon of the optimization problem
            auto        (bool): Flag to set the optimization for autonomous systems
            propagation   (str): Method for state propagation ('exp' for matrix exponential, 'int' for numerical integration)
            plot         (str): Plotting method ('display', 'save', 'none')
            
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
        
        # Define the system's variables
        self.x = []
        # Control input defined as a list of parameters (will be set during optimization)
        self.u = []
        for i in range(self.n_phases):
            if self.auto:
                self.n_inputs = 0
                self.u.append(None)
            else:
                # Initialize with zeros, NO requires_grad during precomputation
                self.u.append(torch.zeros(self.n_inputs, dtype=torch.float64, requires_grad=False))
        
        # Phase duration as parameters (will be set during optimization)
        # Initialize with time_horizon/n_phases as starting guess, NO requires_grad during precomputation
        self.delta = [torch.tensor(self.time_horizon / self.n_phases, dtype=torch.float64, requires_grad=False) for i in range(self.n_phases)]
        
        # Initialize the matrices
        self.E = []
        self.phi_f = []
        self.autonomous_evol = []
        self.forced_evol = []
        self.evol = []
        self.H = []
        self.J = []
        self.L = []
        self.M = []
        self.R = []
        self.S = []
        self.Sr = []
        self.C = []
        self.N = []
        self.D = []
        self.G = []
        self.x_opt = []
        self.S_num = []
        self.S_int = []
        self.Sr_int = []
    
    def load_model(self, model) -> None:
        """
        Load the switched linear model 
        
        Args:
            model   (struct): Structure that stores all the informations about the switched linear model
        """
        A = []
        B = []
        for i in range(self.n_phases):
            id = i % len(model['A'])
            A.append(model['A'][id])
            # Check if the input matrix is empty
            if model['B']:
                B.append(model['B'][id])
            else:
                B.append(np.zeros((model['A'][id].shape[0], 1)))
              
        self.A = A
        self.B = B
        
        # Define the number of modes of the system
        self.n_modes = len(A)
        
    def integrator(self, func, t0, tf, *args):
        """
        Integrates f(t) between t0 and tf using the given function func using the composite Simpson's 1/3 rule.
        
        Args:
            func    (callable): The function that describes the system dynamics
            t0      (float or tensor): The initial time
            tf      (tensor): The final time
            *args   (tensor): Additional arguments to pass to the function
            
        Returns:
            integral    (tensor): The result of the integration
        """
        # Number of steps for the integration
        steps = 10
        
        # Check if args is not empty and set the input accordingly
        input_arg = args[0] if args else None

        # Integration using the composite Simpson's 1/3 rule
        h = (tf - t0) / steps
        if isinstance(t0, torch.Tensor):
            t = t0.clone()
        else:
            t = torch.tensor(t0, dtype=tf.dtype, device=tf.device if isinstance(tf, torch.Tensor) else 'cpu')

        # Determine integration mode:
        # 1. 'auto' flag means autonomous system - func takes only time argument
        # 2. Tensor input means non-autonomous - func takes (time, input) arguments  
        # 3. No args means simple integration - func takes only time argument
        is_autonomous = input_arg == 'auto'
        is_tensor_input = input_arg is not None and isinstance(input_arg, torch.Tensor)

        # Integration for different cases
        if is_autonomous or input_arg is None:
            # Autonomous or simple integration: func(t)
            S = func(t) + func(tf)
        else:
            # Non-autonomous with tensor input: func(t, input)
            S = func(t, input_arg) + func(tf, input_arg)
    
        for k in range(1, steps):
            coefficient = 2 if k % 2 == 0 else 4
            t = t + h
            if is_autonomous or input_arg is None:
                # Autonomous or simple integration: func(t)
                S = S + func(t) * coefficient
            else:
                # Non-autonomous with tensor input: func(t, input)
                S = S + func(t, input_arg) * coefficient

        integral = S * (h / 3)
        
        return integral
    
    def compute_integral(self, A, B, tmin, tmax):
        """
        Computes the forced evolution of the system's state using CasADi for symbolic operations.
        
        Args:
        A (numpy.ndarray): The system matrix.
        B (numpy.ndarray): The input matrix.
        tmin (float): The start time for the integration.
        tmax (float): The end time for the integration.
        
        Returns:
        numpy.ndarray: The result of the integral computation.
        """
        
        # # Following "Computing Integrals Involving the Matrix Exponential" by C. F. Van Loan
        # # We compute the system's state evolution using the matrix exponential
        
        # dyn1 = np.hstack((A, B))
        # dyn2 = np.zeros((B.shape[1], A.shape[0]+B.shape[1]))
        # dyn = np.vstack((dyn1, dyn2))
        
        # # Compute the integral as a matrix exponential
        # expm_dyn = self.expm(dyn, tmax-tmin)
        # integral_result = expm_dyn[:A.shape[0], A.shape[0]:]
        
        # Define the function to be integrated
        def integrand(s):
            return self.expm(A, tmax - s) @ B
    
        integral_result = self.integrator(integrand, tmin, tmax)
        
        return integral_result
    
    def expm(self, A, delta):
        """
        Computes the matrix exponential of A[index] * delta_i using CasADi.
        
        Args:
            A (np.array): "A" matrix the mode for which to compute the matrix exponential.
            delta (ca.SX): time variable
            
        Returns:
            exp_max (ca.SX): The computed matrix exponential.
        """        
        
        n = A.shape[0]  # Size of matrix A
        # Convert A to tensor if it's a numpy array
        if isinstance(A, np.ndarray):
            A_tensor = torch.tensor(A, dtype=torch.float64, requires_grad=False)
        else:
            A_tensor = A
            
        # Determine device and dtype from delta if it's a tensor
        if isinstance(delta, torch.Tensor):
            device = delta.device
            dtype = delta.dtype
            result = torch.eye(n, dtype=dtype, device=device, requires_grad=False)
        else:
            device = 'cpu'
            dtype = torch.float64
            result = torch.eye(n, dtype=dtype, requires_grad=False)
            delta = torch.tensor(delta, dtype=dtype, requires_grad=False)
        
        # Ensure A_tensor is on the right device and doesn't require grad
        A_tensor = A_tensor.to(device=device, dtype=dtype)
        if A_tensor.requires_grad:
            A_tensor = A_tensor.detach()
        
        # Number of terms for the Taylor series expansion
        # num_terms = self._get_n_terms_expm_approximation()
        # num_terms = 6
        
        # A_power = A_tensor.clone()
        # for k in range(1, num_terms+1):
        #     if k > 1:
        #         A_power = A_power @ A_tensor
        #     term = A_power * (delta ** k) / factorial(k)
        #     result = result + term

        # Use PyTorch's built-in matrix exponential for better performance and accuracy
        result = torch.matrix_exp(A_tensor * delta)
        
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
    
    def _mat_exp_prop_exp(self, index, Q, R):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.

        Returns:
        Ei      (ca.SX): The matrix exponential of Ai*delta_i.
        phi_f_i (ca.SX): The integral part multiplied by the control input ui.
        Hi      (ca.SX): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        Li      (ca.SX): Matrix for the cost function
        Mi      (ca.SX): Matrix for the cost function
        Ri      (ca.SX): Matrix for the cost function
        
        """        
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
                
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Create a big matrix of dimentions (3n+m)x(3n+m) 
        C = np.zeros((3*self.n_states + self.n_inputs, 3*self.n_states + self.n_inputs))
        # Fill the matrix
        C[:self.n_states, :self.n_states] = -np.transpose(A)
        C[:self.n_states, self.n_states:2*self.n_states] = np.eye(self.n_states)
        C[self.n_states:2*self.n_states, self.n_states:2*self.n_states] = -np.transpose(A)
        C[self.n_states:2*self.n_states, 2*self.n_states:3*self.n_states] = Q
        C[2*self.n_states:3*self.n_states, 2*self.n_states:3*self.n_states] = A
        C[2*self.n_states:3*self.n_states, 3*self.n_states:] = B
        
        # Compute matrix exponential
        exp_C = self.expm(C, delta_i)
        
        # Extract the instrumental matrices from the matrix exponential
        F3 = exp_C[2*self.n_states:3*self.n_states, 2*self.n_states:3*self.n_states]
        G2 = exp_C[self.n_states:2*self.n_states, 2*self.n_states:3*self.n_states]
        G3 = exp_C[2*self.n_states:3*self.n_states, 3*self.n_states:]
        H2 = exp_C[self.n_states:2*self.n_states, 3*self.n_states:]
        K1 = exp_C[:self.n_states, 3*self.n_states:]
        
        Ei = F3
        Li = F3.T @ G2        
        
        # Distinct case for autonomous systems
        # Extract the control input
        if self.n_inputs > 0:
            ui = self.u[index]
        
            phi_f_i_ = G3
            
            phi_f_i = phi_f_i_ @ ui

            Mi = F3.T @ H2
            
            # Convert B to tensor if needed
            if isinstance(B, np.ndarray):
                B_tensor = torch.tensor(B, dtype=torch.float64)
            else:
                B_tensor = B
            
            # Ensure tensors are on same device
            if isinstance(F3, torch.Tensor):
                B_tensor = B_tensor.to(F3.device)
                Ri = (B_tensor.T @ F3.T @ K1) + (B_tensor.T @ F3.T @ K1).T
            else:
                # F3 is still numpy, convert to compute Ri
                F3_t = torch.tensor(F3, dtype=torch.float64) if isinstance(F3, np.ndarray) else F3
                K1_t = torch.tensor(K1, dtype=torch.float64) if isinstance(K1, np.ndarray) else K1
                Ri = (B_tensor.T @ F3_t.T @ K1_t) + (B_tensor.T @ F3_t.T @ K1_t).T
            
            # Create the H matrix related to the i-th mode (only for the non-autonomous case)
            Hi = []
            
            # Fill the Hk matrix with the k-th column of phi_f_i_ (integral term)
            for k in range(ui.shape[0]):
                Hk = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float64)
                if isinstance(phi_f_i_, torch.Tensor):
                    Hk[:self.n_states, self.n_states] = phi_f_i_[:, k]
                else:
                    Hk[:self.n_states, self.n_states] = torch.tensor(phi_f_i_[:, k], dtype=torch.float64)
                Hi.append(Hk)
        
            return Ei, phi_f_i, Hi, Li, Mi, Ri
        else:
            return Ei, torch.zeros(0, dtype=torch.float64), [], Li, torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
        
    def _mat_exp_prop_int(self, index):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.

        Returns:
        Ei      (ca.SX): The matrix exponential of Ai*delta_i.
        phi_f_i (ca.SX): The integral part multiplied by the control input ui.
        Hi      (ca.SX): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
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
                Hk = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float64)
                if isinstance(phi_f_i_, torch.Tensor):
                    Hk[:self.n_states, self.n_states] = phi_f_i_[:, k]
                else:
                    Hk[:self.n_states, self.n_states] = torch.tensor(phi_f_i_[:, k], dtype=torch.float64)
                Hi.append(Hk)
        
            return Ei, phi_f_i, Hi, torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
        else:
            return Ei, torch.zeros(0, dtype=torch.float64), 0, torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
        
    def mat_exp_prop(self, index, Q, R):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.

        Returns:
        Ei      (ca.SX): The matrix exponential of Ai*delta_i.
        phi_f_i (ca.SX): The integral part multiplied by the control input ui.
        Hi      (ca.SX): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        Li      (ca.SX): Matrix for the cost function
        Mi      (ca.SX): Matrix for the cost function
        Ri      (ca.SX): Matrix for the cost function
        
        """        
        if self.propagation == 'exp':
            return self._mat_exp_prop_exp(index, Q, R)
        else:
            return self._mat_exp_prop_int(index)
        
    def _propagate_state(self, x0):
        self.x = [x0]
        for i in range(self.n_phases):
            if self.n_inputs > 0:
                self.x.append(self.E[i] @ self.x[i] + self.phi_f[i])
            else:
                self.x.append(self.E[i] @ self.x[i])
        
    def transition_matrix(self, phi_a, phi_f):
        """
        Computes the transition matrix for the given index.
        
        Args:
        phi_a (ca.SX): The matrix exponential of the system matrix.
        phi_f (ca.SX): The integral term.
        
        Returns:
        phi (ca.SX): The transition matrix.
        
        """
        phi = torch.zeros(self.n_states+1, self.n_states+1, dtype=torch.float64)
        
        # Distinct case for autonomous and non-autonomous systems
        if self.n_inputs > 0:
            if isinstance(phi_a, torch.Tensor):
                phi[:self.n_states, :self.n_states] = phi_a
                phi[:self.n_states, self.n_states] = phi_f
            else:
                phi[:self.n_states, :self.n_states] = torch.tensor(phi_a, dtype=torch.float64)
                phi_f_tensor = torch.tensor(phi_f, dtype=torch.float64) if isinstance(phi_f, np.ndarray) else phi_f
                phi[:self.n_states, self.n_states] = phi_f_tensor
            phi[-1, -1] = 1
        else:
            if isinstance(phi_a, torch.Tensor):
                phi[:self.n_states, :self.n_states] = phi_a
            else:
                phi[:self.n_states, :self.n_states] = torch.tensor(phi_a, dtype=torch.float64)
            phi[-1, -1] = 1
        
        return phi
           
    def D_matrix(self, index, Q):
        """
        Computes the D matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (np.array): The weight matrix.
        
        Returns:
        D (ca.SX): The D matrix.
        
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
        
        # Define integrand function for each control input dimension
        for k in range(ui.shape[0]):
            def integrand(eta):
                # Compute the integral term
                phi_a_t = self.expm(A, eta)
                phi_f_t = self.compute_integral(A, B, 0, eta)
                phi_t = self.transition_matrix(phi_a_t, phi_f_t @ ui)
                
                # Create Hij_t matrix
                Hij_t = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float64)
                if isinstance(phi_f_t, torch.Tensor):
                    Hij_t[:self.n_states, self.n_states] = phi_f_t[:, k]
                else:
                    Hij_t[:self.n_states, self.n_states] = torch.tensor(phi_f_t[:, k], dtype=torch.float64)
            
                # Convert Q to tensor if needed
                Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
                
                arg = Hij_t.T @ Q_tensor @ phi_t + phi_t.T @ Q_tensor @ Hij_t
                return 0.5 * arg
            
            # Compute D matrix element
            Dij = self.integrator(integrand, 0, delta_i)
            D.append(Dij)
        
        return D
    
    def _S_matrix_exp(self, index, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):      The index of the mode.
        Q       (np.array): The weight matrix.
        xr      (np.array): The reference state.
        
        Returns:
        S       (ca.SX):    The S matrix.
        Optional:
        Sr      (ca.SX):    The Sr matrix.
        
        """
        # Extract the autonomous and non-autonomous parts of the state
        phi_a = self.E[index]
        phi_f = self.phi_f[index]
        
        # Extract the input (if present)
        if self.n_inputs > 0:
            ui = self.u[index]
        
        # Extract the matrices for the integral term
        Li = self.L[index]
        Mi = self.M[index]
        Ri = self.R[index]
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[0]
        
        phi_i = self.transition_matrix(phi_a, phi_f)
        
        S_int = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float64)
        
        # Convert to tensors if needed
        Li_tensor = torch.tensor(Li, dtype=torch.float64) if isinstance(Li, np.ndarray) else Li
        S_int[:self.n_states, :self.n_states] = Li_tensor
        
        if self.n_inputs > 0:
            Mi_tensor = torch.tensor(Mi, dtype=torch.float64) if isinstance(Mi, np.ndarray) else Mi
            Ri_tensor = torch.tensor(Ri, dtype=torch.float64) if isinstance(Ri, np.ndarray) else Ri
            
            # Compute Mi @ ui and ensure it's a column vector
            Mi_ui = (Mi_tensor @ ui).reshape(-1, 1)
            
            S_int[:self.n_states, self.n_states:] = Mi_ui
            S_int[self.n_states:, :self.n_states] = Mi_ui.T
            S_int[self.n_states:, self.n_states:] = (ui @ Ri_tensor @ ui).reshape(1, 1)
        
        # If a reference state is given, compute both the Sr matrix and the S matrix
        if xr is not None:
            S = S_int + phi_i.T @ S_prev @ phi_i
            return S
        
        # Compute S matrix
        S = S_int + phi_i.T @ S_prev @ phi_i
        
        return S
    
    def _S_matrix_int(self, index, Q, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):      The index of the mode.
        Q       (np.array): The weight matrix.
        xr      (np.array): The reference state.
        
        Returns:
        S       (ca.SX):    The S matrix.
        Optional:
        Sr      (ca.SX):    The Sr matrix.
        
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
        
        # Define integrand function for S matrix computation
        if self.n_inputs == 0:
            def integrand_S(eta):
                phi_a_t = self.expm(A, eta)
                phi_f_t = self.compute_integral(A, B, 0, eta)
                phi_t = self.transition_matrix(phi_a_t, phi_f_t)
                Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
                return phi_t.T @ Q_tensor @ phi_t
            
            S_int = self.integrator(integrand_S, 0, delta_i, 'auto')
            self.S_int.append(lambda deltas: 0.5 * S_int)
            
            if xr is not None:
                def integrand_Sr(eta):
                    phi_a_t = self.expm(A, eta)
                    phi_f_t = self.compute_integral(A, B, 0, eta)
                    phi_t = self.transition_matrix(phi_a_t, phi_f_t)
                    Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
                    xr_tensor = torch.tensor(xr, dtype=torch.float64) if isinstance(xr, np.ndarray) else xr
                    return phi_t.T @ Q_tensor @ xr_tensor
                
                Sr_int = self.integrator(integrand_Sr, 0, delta_i, 'auto')
                self.Sr_int.append(lambda deltas: Sr_int)
        else:
            def integrand_S(eta):
                phi_a_t = self.expm(A, eta)
                phi_f_t = self.compute_integral(A, B, 0, eta)
                phi_t = self.transition_matrix(phi_a_t, phi_f_t @ ui)
                Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
                return phi_t.T @ Q_tensor @ phi_t
            
            S_int = self.integrator(integrand_S, 0, delta_i, ui)
            self.S_int.append(lambda delta, u: 0.5 * S_int)
            
            if xr is not None:
                def integrand_Sr(eta):
                    phi_a_t = self.expm(A, eta)
                    phi_f_t = self.compute_integral(A, B, 0, eta)
                    phi_t = self.transition_matrix(phi_a_t, phi_f_t @ ui)
                    Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
                    xr_tensor = torch.tensor(xr, dtype=torch.float64) if isinstance(xr, np.ndarray) else xr
                    return phi_t.T @ Q_tensor @ xr_tensor
                
                Sr_int = self.integrator(integrand_Sr, 0, delta_i, ui)
                self.Sr_int.append(lambda deltas, us: 0.5 * Sr_int)
        
        phi_i = self.transition_matrix(phi_a, phi_f)
        
        # If a reference state is given, compute both the Sr matrix and the S matrix
        if xr is not None:
            Sr = Sr_int + phi_i.T @ Sr_prev
            S = S_int + phi_i.T @ S_prev @ phi_i
            return S, Sr
        
        # Compute S matrix
        S = S_int + phi_i.T @ S_prev @ phi_i
        
        return S    
    
    def S_matrix(self, index, Q, xr=None):
        """
        Computes the S matrix for the given index.
        If a reference state is given, it computes the Sr matrix in order to minimize the error
        between the reference state and the state trajectory.
        
        Args:
        index   (int):      The index of the mode.
        Q       (np.array): The weight matrix.
        xr      (np.array): The reference state.
        
        Returns:
        S       (ca.SX):    The S matrix.
        Optional:
        Sr      (ca.SX):    The Sr matrix.
        
        """
        if self.propagation == 'exp':
            return self._S_matrix_exp(index, xr)
        else:
            return self._S_matrix_int(index, Q, xr)
        
    def C_matrix(self, index, Q):
        """
        Computes the C matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (np.array): The weight matrix.
        
        Returns:
        C (ca.SX): The C matrix.
        
        """
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Define the M matrix
        M = torch.zeros(self.n_states + 1, self.n_states + 1, dtype=torch.float64)
        
        A_tensor = torch.tensor(A, dtype=torch.float64) if isinstance(A, np.ndarray) else A
        B_tensor = torch.tensor(B, dtype=torch.float64) if isinstance(B, np.ndarray) else B
        
        M[:self.n_states, :self.n_states] = A_tensor
        M[:self.n_states, self.n_states] = B_tensor @ ui
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index+1]
        
        Q_tensor = torch.tensor(Q, dtype=torch.float64) if isinstance(Q, np.ndarray) else Q
        C = 0.5*Q_tensor + M.T @ S_prev + S_prev @ M
        
        return C
           
    def N_matrix(self, index):
        """
        Computes the N matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        
        Returns:
        N (ca.SX): The N matrix.
        
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
        
    def G_matrix(self, R):
        """
        Computes the G matrix.
        
        Args:
        R (np.array): The weight matrix.
        
        Returns:
        G (ca.SX): The G matrix.
        
        """
        
        G = 0
        R_tensor = torch.tensor(R, dtype=torch.float64) if isinstance(R, np.ndarray) else R
        
        for i in range(self.n_phases):
            # Use @ for matrix multiplication, which handles 1D tensors correctly
            pippo = 0.5 * (self.u[i] @ R_tensor @ self.u[i]) * self.delta[i]
            G = G + pippo
            
        return G
        
    def cost_function(self, R, x0=None, sym_x0=False):
        """
        Computes the cost function.
        
        Args:
        R (np.array): The weight matrix.
        x0 (np.array): The initial state.
        sym_x0 (bool): Flag to indicate if x0 is symbolic.
        Optional:
            xr (np.array): The reference state.
            E (np.array): The weight matrix for the terminal state.
        
        Returns:
        J (ca.Function): The cost function.
        
        """
        # Check if x0 is symbolic (we'll handle this differently in PyTorch)
        if sym_x0:
            # Return a lambda function that computes cost given x0
            def cost(*args):
                # Last argument is x0, others are u and delta
                if self.n_inputs == 0:
                    # args are: *delta, x0
                    delta_vals = args[:-1]
                    x0_val = args[-1]
                else:
                    # args are: *u, *delta, x0
                    # u_vals is a list of n_phases tensors, each of shape (n_inputs,)
                    u_vals = list(args[:self.n_phases])
                    delta_vals = list(args[self.n_phases:self.n_phases + self.n_phases])
                    x0_val = args[-1]
                
                # Ensure x0 is a tensor
                if not isinstance(x0_val, torch.Tensor):
                    x0_tensor = torch.tensor(x0_val, dtype=torch.float64)
                else:
                    x0_tensor = x0_val
                
                # Get device from x0_tensor
                device = x0_tensor.device if isinstance(x0_tensor, torch.Tensor) else 'cpu'
                
                # Ensure all tensors are on the same device
                x0_flat = x0_tensor.flatten().to(device)
                ones_tensor = torch.ones(1, dtype=torch.float64, device=device)
                x0_aug = torch.cat([x0_flat, ones_tensor])
                
                S0_tensor = torch.tensor(self.S[0], dtype=torch.float64, device=device) if isinstance(self.S[0], np.ndarray) else self.S[0].to(device)
                J = 0.5 * x0_aug @ S0_tensor @ x0_aug
                
                if self.n_inputs > 0:
                    # Compute G matrix using the provided u and delta values, not self.u and self.delta
                    G = 0
                    R_tensor = torch.tensor(R, dtype=torch.float64, device=device) if isinstance(R, np.ndarray) else R.to(dtype=torch.float64, device=device)
                    
                    for i in range(self.n_phases):
                        # Use the u and delta values from the arguments, ensure correct dtype
                        if isinstance(u_vals[i], torch.Tensor):
                            u_i = u_vals[i].to(dtype=torch.float64, device=device)
                        else:
                            u_i = torch.tensor(u_vals[i], dtype=torch.float64, device=device)
                            
                        if isinstance(delta_vals[i], torch.Tensor):
                            delta_i = delta_vals[i].to(dtype=torch.float64, device=device)
                        else:
                            delta_i = torch.tensor(delta_vals[i], dtype=torch.float64, device=device)
                            
                        pippo = 0.5 * (u_i @ R_tensor @ u_i) * delta_i
                        G = G + pippo
                    
                    J = J + 0.5 * G
                    
                return J
                
        else:
            # Compute the cost function with numeric x0
            x0 = np.reshape(x0, (-1, 1))
            x0_tensor = torch.tensor(x0, dtype=torch.float64)
            S0_tensor = torch.tensor(self.S[0], dtype=torch.float64) if isinstance(self.S[0], np.ndarray) else self.S[0]
            J = 0.5 * x0_tensor.T @ S0_tensor @ x0_tensor
            
            if self.n_inputs > 0:
                J = J + 0.5 * self.G_matrix(R)
            
            # Return a lambda function that computes cost given u and delta
            def cost(*args):
                # Return the precomputed J (symbolic in terms of u and delta)
                return J
            
        return cost
        
    def grad_cost_function(self, index, R):
        """
        Computes the gradient of the cost function.
        
        Args:
        index (int): The index of the mode.
        R (np.array): The weight matrix.
        
        Returns:
        du (ca.SX): The gradient of the cost function with respect to the control input.
        d_delta (ca.SX): The gradient of the cost function with respect to the phase duration.
        
        """
        
        # Create the augmented state vectors
        x_aug = torch.zeros(self.n_states + 1, dtype=torch.float64)
        x_next_aug = torch.zeros(self.n_states + 1, dtype=torch.float64)
        
        # Convert x to tensor if needed
        x_i = self.x[index]
        x_next = self.x[index+1]
        
        if isinstance(x_i, torch.Tensor):
            x_aug[:self.n_states] = x_i
        else:
            x_aug[:self.n_states] = torch.tensor(x_i, dtype=torch.float64).flatten()
        x_aug[self.n_states] = 1
        
        if isinstance(x_next, torch.Tensor):
            x_next_aug[:self.n_states] = x_next
        else:
            x_next_aug[:self.n_states] = torch.tensor(x_next, dtype=torch.float64).flatten()
        x_next_aug[self.n_states] = 1
        
        R_tensor = torch.tensor(R, dtype=torch.float64) if isinstance(R, np.ndarray) else R

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

            term_r = ui[j] * R_tensor[j, j] * delta_i
            term_d = x_aug @ Dij @ x_aug if Dij is not None else 0
            term_n = x_next_aug @ Nij @ x_next_aug if Nij is not None else 0

            du_j = term_r + term_d + term_n
            du.append(du_j)

        # Compute the gradient of the cost function with respect to the phase duration
        d_delta = 0.5 * (ui @ R_tensor @ ui) + x_next_aug @ C @ x_next_aug
        
        return du, d_delta
    
    def precompute_matrices(self, x0, Q, R, E, xr=None) -> None:
        """
        Precomputes the matrices that are necessary to write the cost function and its gradient.
        
        Args:
        x0  (np.array): The initial state.
        Q   (np.array): The weight matrix for the state.
        R   (np.array): The weight matrix for the control.
        E   (np.array): The weight matrix for the terminal state.
        xr  (np.array): The reference state.
        """  
        # Augment the weight matrices
        Q_ = torch.tensor(block_diag(Q, 0), dtype=torch.float64)
        E_ = torch.tensor(block_diag(E, 0), dtype=torch.float64)
        
        for i in range(self.n_phases):
            # Compute the matrix exponential properties
            Ei, phi_f_i, Hi, Li, Mi, Ri = self.mat_exp_prop(i, Q, R)
            self.E.append(Ei)
            # Store lambda functions instead of CasADi functions
            self.autonomous_evol.append(lambda delta: Ei)
            self.phi_f.append(phi_f_i)
            self.forced_evol.append(lambda u, delta: phi_f_i)
            self.H.append(Hi)
            self.L.append(Li)
            self.M.append(Mi)
            self.R.append(Ri)
        
            if self.n_inputs > 0:
                # Compute the D matrix
                D = self.D_matrix(i, Q_)
                self.D.append(D)
            
                # Compute the G matrix
                G = self.G_matrix(R)
                self.G.append(G)
        
        # Initialize the S matrix with the terminal cost (if needed)
        self.S.append(0.5*E_)
        if xr is not None:
            xr_aug = np.append(xr, 1)
            self.Sr.append(0.5*E_@ xr_aug)

        for i in range(self.n_phases-1, -1, -1):
            if xr is not None:
                # Compute the S and Sr matrices
                S, Sr = self.S_matrix(i, Q_)
                self.S.insert(0, S)
                # self.Sr.insert(0, Sr)
            else:
                # Compute the S matrix
                S = self.S_matrix(i, Q_)
                self.S.insert(0, S)
            
        #     # Create the S_num function for debugging
        #     if self.n_inputs == 0:
        #         S_num = ca.Function('S_num', [*self.delta], [S])
        #     else:
        #         S_num = ca.Function('S_num', [*self.delta, *self.u], [S])
                
        #     self.S_num.insert(0, S_num)
        
        # Compute the C and N matrices
        for i in range(self.n_phases):
            C = self.C_matrix(i, Q_)
            self.C.append(C)
            if self.n_inputs > 0:
                N = self.N_matrix(i)
                self.N.append(N)
                
        # Propagate the state using the computed matrices.
        self._propagate_state(x0)
       
    def state_extraction(self, delta_opt, *args):
        """
        Extract the optimal values of the state trajectory based on the optimized values of u and delta
        """    
        
        # Check if args is not empty and set the input accordingly
        u_opt = args[0] if args else None   
        
        x_opt = []
        for i in range(self.n_phases+1):
            # Get the state directly (already computed during precomputation)
            x_i = self.x[i]
            
            # Convert to numpy array for output
            if isinstance(x_i, torch.Tensor):
                x_opt.append(x_i.detach().cpu().numpy())
            elif isinstance(x_i, np.ndarray):
                x_opt.append(x_i)
            else:
                # It's a list or similar, convert
                x_opt.append(np.array(x_i))
                
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
            
            x_opt_num = np.array([x_opt[i].elements() for i in range(len(x_opt))])
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
            # ax.scatter(tgrid[::M], x_opt_num[:, i])
        ax.set_xlim([0, self.time_horizon])
        # Add a legend
        ax.legend(loc='upper right')
        # Add vertical lines to identify phase changes instants
        time = 0
        for i in range(self.n_phases):
            time = time + delta_opt[i]
            plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)
        ax.set(xlabel='Time', ylabel='State')
        # ax.grid()
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
                    # ax[i].grid()
                    ax[i].set_xlim([0, self.time_horizon])
                    # Add vertical lines to identify phase changes instants
                    time = 0
                    for j in range(self.n_phases):
                        time = time + delta_opt[j]
                        ax[i].axvline(x=time, color='k', linestyle='--', linewidth=0.5)
                        # plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)
            else:
                ax.step(tgrid[::points], np.array(u_opt_list), where='post', linewidth=2)
                ax.set(xlabel='Time', ylabel='Input')
                ax.set_xlim([0, self.time_horizon])
                # ax.grid()
                # Add vertical lines to identify phase changes instants
                time = 0
                for i in range(self.n_phases):
                    time = time + delta_opt[i]
                    plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)   
        
        if self.plot == 'save':
            plt.savefig(filename + '_optimal_input.pdf', format='pdf', bbox_inches='tight')
        elif self.plot == 'display':
            plt.show()
