"""
PyTorch implementation of optimal control for switched linear control systems.

This module implements the optimization framework for switched linear systems
using automatic differentiation and PyTorch's optimization capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator


class SwitchedLinearSystem:
    """
    Optimal control solver for switched linear control systems.
    
    The system dynamics are:
        dx/dt = A_i(t) * x(t) + B_i(t) * u(t)
    
    where i(t) switches between different modes according to a switching sequence.
    """
    
    def __init__(
        self,
        A_matrices: List[torch.Tensor],
        B_matrices: List[torch.Tensor],
        Q: torch.Tensor,
        R: torch.Tensor,
        En: torch.Tensor,
        x0: torch.Tensor,
        T: float,
        N: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize the switched linear system.
        
        Args:
            A_matrices: List of state matrices for each mode [Ns x Nx x Nx]
            B_matrices: List of input matrices for each mode [Ns x Nx x Nu]
            Q: State cost matrix [Nx x Nx]
            R: Control cost matrix [Nu x Nu]
            En: Terminal state cost matrix [Nx x Nx]
            x0: Initial state [Nx]
            T: Total time horizon
            N: Number of phases/switches
            device: torch device ('cpu' or 'cuda')
            dtype: torch data type
        """
        self.device = device
        self.dtype = dtype
        
        # System parameters
        self.Ns = len(A_matrices)  # Number of systems/modes
        self.N = N  # Number of phases
        self.T = T  # Time horizon
        
        # Convert to tensors
        self.A = [A.to(device=device, dtype=dtype) for A in A_matrices]
        self.B = [B.to(device=device, dtype=dtype) for B in B_matrices]
        self.Q = Q.to(device=device, dtype=dtype)
        self.R = R.to(device=device, dtype=dtype)
        self.En = En.to(device=device, dtype=dtype)
        self.x0 = x0.to(device=device, dtype=dtype)
        
        # Dimensions
        self.Nx = self.A[0].shape[0]  # Number of states
        self.Nu = self.B[0].shape[1]  # Number of control inputs
        
        # Augmented cost matrix Q_bar = [Q, 0; 0, 0]
        self.Q_bar = torch.zeros(self.Nx + 1, self.Nx + 1, device=device, dtype=dtype)
        self.Q_bar[:self.Nx, :self.Nx] = self.Q
        
    def matrix_exponential(self, A: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Compute matrix exponential exp(A * delta)."""
        return torch.linalg.matrix_exp(A * delta)
    
    def compute_integral(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Compute integral: ∫_0^delta exp(A*(delta-s)) * B ds
        
        Uses numerical integration with trapezoidal rule.
        """
        s_vals = torch.linspace(0, delta.item(), num_steps, device=self.device, dtype=self.dtype)
        ds = delta / (num_steps - 1)
        
        integral = torch.zeros_like(B)
        for i, s in enumerate(s_vals):
            exp_term = self.matrix_exponential(A, delta - s)
            weight = 1.0 if (i == 0 or i == len(s_vals) - 1) else 2.0
            integral += weight * (exp_term @ B)
        
        integral *= ds / 2.0
        return integral
    
    def transition_matrix(self, phi_a: torch.Tensor, phi_f: torch.Tensor) -> torch.Tensor:
        """
        Create augmented transition matrix.
        
        Returns:
            [[phi_a, phi_f],
             [0,      1    ]]
        """
        n = phi_a.shape[0]
        phi = torch.zeros(n + 1, n + 1, device=self.device, dtype=self.dtype)
        phi[:n, :n] = phi_a
        phi[:n, n:n+1] = phi_f
        phi[n, n] = 1.0
        return phi
    
    def mat_exp_prop(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Compute state propagation using matrix exponential.
        
        Returns:
            x_next: Next state
            E: exp(A * delta)
            phi_f: Forced response term
            H: List of sensitivity matrices
        """
        E = self.matrix_exponential(A, delta)
        phi_f_mat = self.compute_integral(A, B, delta)
        phi_f = phi_f_mat @ u
        x_next = E @ x + phi_f
        
        # Compute H matrices
        H = []
        for k in range(self.Nu):
            H_k = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
            H_k[:self.Nx, self.Nx] = phi_f_mat[:, k]
            H.append(H_k)
        
        return x_next, E, phi_f, H
    
    def compute_S(
        self,
        S_next: torch.Tensor,
        E: torch.Tensor,
        phi_f: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute S matrix (backward propagation of cost-to-go).
        
        S_i = ∫_0^delta Φ(t)' * Q_bar * Φ(t) dt + Φ(delta)' * S_{i+1} * Φ(delta)
        """
        # Numerical integration for the integral term
        eta_vals = torch.linspace(0, delta.item(), num_steps, device=self.device, dtype=self.dtype)
        d_eta = delta / (num_steps - 1)
        
        S_int = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
        
        for i, eta in enumerate(eta_vals):
            phi_a_t = self.matrix_exponential(A, eta)
            phi_f_t = self.compute_integral(A, B, eta)
            phi_t = self.transition_matrix(phi_a_t, phi_f_t @ u)
            
            weight = 1.0 if (i == 0 or i == len(eta_vals) - 1) else 2.0
            S_int += weight * (phi_t.T @ self.Q_bar @ phi_t)
        
        S_int *= d_eta / 2.0
        
        # Add terminal cost term
        phi_i = self.transition_matrix(E, phi_f)
        S = S_int + phi_i.T @ S_next @ phi_i
        
        return S
    
    def compute_C(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        u: torch.Tensor,
        S_next: torch.Tensor
    ) -> torch.Tensor:
        """Compute C matrix for gradient calculation."""
        M = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
        M[:self.Nx, :self.Nx] = A
        M[:self.Nx, self.Nx:self.Nx+1] = B @ u
        
        C = self.Q_bar + M.T @ S_next + S_next @ M
        return C
    
    def compute_D(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor,
        num_steps: int = 50
    ) -> List[torch.Tensor]:
        """Compute D matrices for gradient calculation."""
        eta_vals = torch.linspace(0, delta.item(), num_steps, device=self.device, dtype=self.dtype)
        d_eta = delta / (num_steps - 1)
        
        D = []
        for k in range(self.Nu):
            D_k = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
            
            for i, eta in enumerate(eta_vals):
                phi_a_t = self.matrix_exponential(A, eta)
                phi_f_t = self.compute_integral(A, B, eta)
                phi_t = self.transition_matrix(phi_a_t, phi_f_t @ u)
                
                H_t = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
                H_t[:self.Nx, self.Nx] = phi_f_t[:, k]
                
                weight = 1.0 if (i == 0 or i == len(eta_vals) - 1) else 2.0
                arg = H_t.T @ self.Q_bar @ phi_t + phi_t.T @ self.Q_bar @ H_t
                D_k += weight * arg
            
            D_k *= d_eta / 2.0
            D.append(D_k)
        
        return D
    
    def compute_N(self, H: torch.Tensor, S_next: torch.Tensor) -> torch.Tensor:
        """Compute N matrix."""
        return H.T @ S_next + S_next @ H
    
    def compute_G(self, u_all: torch.Tensor, delta_all: torch.Tensor) -> torch.Tensor:
        """Compute control cost term G = Σ u_k' * R * u_k * delta_k."""
        G = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for k in range(self.N):
            u_k = u_all[k * self.Nu:(k + 1) * self.Nu]
            G += u_k @ self.R @ u_k * delta_all[k]
        return G
    
    def forward_propagate(
        self,
        u_all: torch.Tensor,
        delta_all: torch.Tensor
    ) -> Tuple[torch.Tensor, List, List, List, List]:
        """
        Forward propagate the states through all phases.
        
        Returns:
            x_all: All states [Nx x (N+1)]
            E_all: All E matrices
            phi_f_all: All phi_f vectors
            H_all: All H matrices
            Phi_all: All transition matrices
        """
        x_all = torch.zeros(self.Nx, self.N + 1, device=self.device, dtype=self.dtype)
        x_all[:, 0] = self.x0
        
        E_all = []
        phi_f_all = []
        H_all = []
        Phi_all = []
        
        for k in range(self.N):
            i = k % self.Ns  # Mode index
            u_k = u_all[k * self.Nu:(k + 1) * self.Nu]
            delta_k = delta_all[k]
            
            x_next, E, phi_f, H = self.mat_exp_prop(
                self.A[i], self.B[i], x_all[:, k], u_k, delta_k
            )
            
            x_all[:, k + 1] = x_next
            E_all.append(E)
            phi_f_all.append(phi_f)
            H_all.append(H)
            Phi_all.append(self.transition_matrix(E, phi_f))
        
        return x_all, E_all, phi_f_all, H_all, Phi_all
    
    def backward_propagate(
        self,
        x_all: torch.Tensor,
        u_all: torch.Tensor,
        delta_all: torch.Tensor,
        E_all: List,
        phi_f_all: List,
        H_all: List
    ) -> Tuple[List, List, List, List]:
        """
        Backward propagate the S matrices and compute gradient components.
        
        Returns:
            S_all: S matrices
            C_all: C matrices
            D_all: D matrices
            N_all: N matrices
        """
        S_all = [None] * (self.N + 1)
        S_all[self.N] = torch.zeros(self.Nx + 1, self.Nx + 1, device=self.device, dtype=self.dtype)
        S_all[self.N][:self.Nx, :self.Nx] = self.En
        
        C_all = []
        D_all = []
        N_all = []
        
        for k in range(self.N - 1, -1, -1):
            j = k % self.Ns
            u_k = u_all[k * self.Nu:(k + 1) * self.Nu]
            delta_k = delta_all[k]
            
            S_k = self.compute_S(
                S_all[k + 1], E_all[k], phi_f_all[k],
                self.A[j], self.B[j], u_k, delta_k
            )
            S_all[k] = S_k
            
            C_k = self.compute_C(self.A[j], self.B[j], u_k, S_all[k + 1])
            C_all.insert(0, C_k)
            
            D_k = self.compute_D(self.A[j], self.B[j], u_k, delta_k)
            D_all.insert(0, D_k)
            
            N_k = [self.compute_N(H_all[k][m], S_all[k + 1]) for m in range(self.Nu)]
            N_all.insert(0, N_k)
        
        return S_all, C_all, D_all, N_all
    
    def cost_function(
        self,
        u_all: torch.Tensor,
        delta_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total cost J.
        
        J = G0 + [x0; 1]' * S_0 * [x0; 1]
        """
        # Forward propagation
        x_all, E_all, phi_f_all, H_all, _ = self.forward_propagate(u_all, delta_all)
        
        # Backward propagation
        S_all, _, _, _ = self.backward_propagate(
            x_all, u_all, delta_all, E_all, phi_f_all, H_all
        )
        
        # Control cost
        G0 = self.compute_G(u_all, delta_all)
        
        # Initial state cost
        x0_aug = torch.cat([self.x0, torch.ones(1, device=self.device, dtype=self.dtype)])
        J = G0 + x0_aug @ S_all[0] @ x0_aug
        
        return J
    
    def gradient_J(
        self,
        x: torch.Tensor,
        x_next: torch.Tensor,
        C: torch.Tensor,
        u: torch.Tensor,
        delta: torch.Tensor,
        D: List[torch.Tensor],
        N: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of cost with respect to u and delta.
        
        Returns:
            d_u: Gradient w.r.t. control [Nu]
            d_delta: Gradient w.r.t. time duration [1]
        """
        x_aug = torch.cat([x, torch.ones(1, device=self.device, dtype=self.dtype)])
        x_next_aug = torch.cat([x_next, torch.ones(1, device=self.device, dtype=self.dtype)])
        
        d_u = torch.zeros(self.Nu, device=self.device, dtype=self.dtype)
        for j in range(self.Nu):
            d_u[j] = (
                2 * u[j] * self.R[j, j] * delta +
                x_aug @ D[j] @ x_aug +
                x_next_aug @ N[j] @ x_next_aug
            )
        
        d_delta = u @ self.R @ u + x_next_aug @ C @ x_next_aug
        
        return d_u, d_delta
    
    def compute_gradient(
        self,
        u_all: torch.Tensor,
        delta_all: torch.Tensor
    ) -> torch.Tensor:
        """Compute full gradient of cost function."""
        # Forward propagation
        x_all, E_all, phi_f_all, H_all, _ = self.forward_propagate(u_all, delta_all)
        
        # Backward propagation
        _, C_all, D_all, N_all = self.backward_propagate(
            x_all, u_all, delta_all, E_all, phi_f_all, H_all
        )
        
        # Compute gradients
        grad_u = []
        grad_delta = []
        
        for k in range(self.N):
            u_k = u_all[k * self.Nu:(k + 1) * self.Nu]
            delta_k = delta_all[k]
            
            d_u, d_delta = self.gradient_J(
                x_all[:, k], x_all[:, k + 1], C_all[k],
                u_k, delta_k, D_all[k], N_all[k]
            )
            
            grad_u.append(d_u)
            grad_delta.append(d_delta)
        
        grad_u = torch.cat(grad_u)
        grad_delta = torch.stack(grad_delta)
        
        return torch.cat([grad_u, grad_delta])
    
    def optimize(
        self,
        u0: Optional[np.ndarray] = None,
        delta0: Optional[np.ndarray] = None,
        x_r: Optional[torch.Tensor] = None,
        max_iter: int = 1000,
        tol: float = 1e-15,
        verbose: bool = True
    ) -> Dict:
        """
        Optimize control inputs and phase durations.
        
        Args:
            u0: Initial guess for controls [N*Nu]
            delta0: Initial guess for phase durations [N]
            x_r: Reference final state (if None, no terminal constraint)
            max_iter: Maximum optimization iterations
            tol: Optimization tolerance
            verbose: Print optimization progress
            
        Returns:
            Dictionary with optimization results
        """
        # Default initial guess
        if u0 is None:
            u0 = np.zeros(self.N * self.Nu)
        if delta0 is None:
            delta0 = np.ones(self.N) * (self.T / self.N)
        
        w0 = np.concatenate([u0, delta0])
        
        # Define objective and gradient for scipy
        def objective(w):
            u_torch = torch.tensor(w[:self.N * self.Nu], device=self.device, dtype=self.dtype)
            delta_torch = torch.tensor(w[self.N * self.Nu:], device=self.device, dtype=self.dtype)
            
            J = self.cost_function(u_torch, delta_torch)
            grad = self.compute_gradient(u_torch, delta_torch)
            
            return J.cpu().numpy(), grad.cpu().numpy()
        
        # Constraints
        constraints = []
        
        # Time constraint: sum(delta) = T
        def time_constraint(w):
            return np.sum(w[self.N * self.Nu:]) - self.T
        
        constraints.append({
            'type': 'eq',
            'fun': time_constraint
        })
        
        # Terminal state constraint (if provided)
        if x_r is not None:
            def terminal_constraint(w):
                u_torch = torch.tensor(w[:self.N * self.Nu], device=self.device, dtype=self.dtype)
                delta_torch = torch.tensor(w[self.N * self.Nu:], device=self.device, dtype=self.dtype)
                x_all, _, _, _, _ = self.forward_propagate(u_torch, delta_torch)
                return (x_all[:, -1] - x_r).cpu().numpy()
            
            constraints.append({
                'type': 'eq',
                'fun': terminal_constraint
            })
        
        # Bounds: delta >= 0
        bounds = [(None, None)] * (self.N * self.Nu) + [(0, None)] * self.N
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': tol, 'disp': verbose}
        )
        
        # Extract results
        u_opt = result.x[:self.N * self.Nu]
        delta_opt = result.x[self.N * self.Nu:]
        tau_opt = np.concatenate([[0], np.cumsum(delta_opt)])
        
        # Compute optimal trajectory
        u_torch = torch.tensor(u_opt, device=self.device, dtype=self.dtype)
        delta_torch = torch.tensor(delta_opt, device=self.device, dtype=self.dtype)
        x_opt, _, _, _, _ = self.forward_propagate(u_torch, delta_torch)
        
        return {
            'u_opt': u_opt,
            'delta_opt': delta_opt,
            'tau_opt': tau_opt,
            'x_opt': x_opt.cpu().numpy(),
            'cost': result.fun,
            'success': result.success,
            'message': result.message,
            'nit': result.nit
        }
    
    def plot_results(self, result: Dict, show: bool = True):
        """Plot optimal trajectory and control."""
        tau_opt = result['tau_opt']
        x_opt = result['x_opt']
        u_opt = result['u_opt']
        
        # Interpolate states
        step = 0.01
        t = np.arange(0, self.T + step, step)
        x_interp = np.zeros((self.Nx, len(t)))
        
        for i in range(self.Nx):
            interpolator = PchipInterpolator(tau_opt, x_opt[i, :])
            x_interp[i, :] = interpolator(t)
        
        # Plot states
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # State trajectory
        ax = axes[0]
        for i in range(self.Nx):
            ax.plot(t, x_interp[i, :], label=f'x{i+1}', linewidth=2)
        
        for tau in tau_opt[1:]:
            ax.axvline(tau, linestyle='--', color='k', alpha=0.3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.set_title('Optimal State Trajectory')
        ax.legend()
        ax.grid(True)
        
        # Control input
        ax = axes[1]
        u_stairs = np.concatenate([u_opt.reshape(self.N, self.Nu), np.full((1, self.Nu), np.nan)])
        
        for j in range(self.Nu):
            ax.step(tau_opt, u_stairs[:, j], where='post', label=f'u{j+1}', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Control')
        ax.set_title('Optimal Control Input')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, axes


def main():
    """Example usage of the SwitchedLinearSystem class."""
    
    # System parameters
    T = 1.0
    Ns = 2
    N = 8
    Nx = 2
    Nu = 1
    
    # Define system matrices
    A_1 = torch.tensor([[0.6, 1.2], [-0.8, 3.4]], dtype=torch.float64)
    B_1 = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    
    A_2 = torch.tensor([[4.0, 3.0], [-1.0, 0.0]], dtype=torch.float64)
    B_2 = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
    
    A_matrices = [A_1, A_2]
    B_matrices = [B_1, B_2]
    
    # Cost matrices
    Q = torch.zeros(Nx, Nx, dtype=torch.float64)
    R = 10.0 * torch.eye(Nu, dtype=torch.float64)
    En = torch.zeros(Nx, Nx, dtype=torch.float64)
    
    # Initial state
    x0 = torch.tensor([10.0, -7.0], dtype=torch.float64)
    
    # Create system
    system = SwitchedLinearSystem(
        A_matrices=A_matrices,
        B_matrices=B_matrices,
        Q=Q,
        R=R,
        En=En,
        x0=x0,
        T=T,
        N=N,
        device='cpu',
        dtype=torch.float64
    )
    
    # Optimize
    print("Optimizing switched linear system...")
    result = system.optimize(verbose=True)
    
    # Print results
    print(f"\nOptimization completed: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Iterations: {result['nit']}")
    print(f"Final cost: {result['cost']:.6e}")
    print(f"\nOptimal controls:\n{result['u_opt']}")
    print(f"\nPhase durations:\n{result['delta_opt']}")
    print(f"\nSwitching times:\n{result['tau_opt']}")
    
    # Plot results
    system.plot_results(result)


if __name__ == "__main__":
    main()
