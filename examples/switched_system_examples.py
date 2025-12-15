"""
Example usage of SwitchedSystemSimulator

This script demonstrates various capabilities of the switched system simulator:
1. State-based switching
2. Time-based switching
3. Controlled vs autonomous systems
4. Different switching strategies
"""

import numpy as np
import sys
import os

from src.switched_system_simulator import SwitchedSystemSimulator


def example_1_state_based_switching():
    """Example 1: Two-mode system with state-based switching."""
    print("\n" + "="*70)
    print("Example 1: State-Based Switching")
    print("="*70)
    
    # Define system modes
    A1 = np.array([[-0.3, 2.0], [-2.0, -0.3]])  # Spiraling outward
    A2 = np.array([[-2.0, -0.5], [0.5, -2.0]])   # Damped
    
    B1 = np.array([[1.0], [0.0]])
    B2 = np.array([[0.0], [0.5]])
    
    # Switching: alternate based on state magnitude
    def switch_cond(x, t, mode):
        norm = np.linalg.norm(x)
        if mode == 0:
            return norm > 0.1  # Switch to damping when too large
        else:
            return norm < 0.05  # Switch back to spiraling when small
    
    sim = SwitchedSystemSimulator(
        A=[A1, A2],
        B=[B1, B2],
        switching_conditions=[switch_cond, switch_cond],
        mode_names=['Spiral', 'Damp']
    )
    
    # Small control input
    control = lambda t: np.array([0.05 * np.sin(0.5 * t)])
    
    x0 = np.array([0.5, 0.2])
    result = sim.simulate(x0, t_span=(0, 30), control=control, dt=0.05)
    
    print(f"Status: {result['message']}")
    print(f"Switches: {result['n_switches']}")
    print(f"Final state: {result['x'][:, -1]}")
    
    sim.plot_trajectory(result, save_path='../images/example1_trajectory.png', show=False)
    sim.plot_phase_portrait(result, save_path='../images/example1_phase.png', show=False)
    print("Plots saved!")


def example_2_time_based_switching():
    """Example 2: Periodic time-based mode switching."""
    print("\n" + "="*70)
    print("Example 2: Time-Based Periodic Switching")
    print("="*70)
    
    # Three different damping modes
    A1 = np.array([[-0.5, 0], [0, -0.5]])
    A2 = np.array([[-1.0, 0], [0, -1.0]])
    A3 = np.array([[-1.5, 0], [0, -1.5]])
    
    # Switch every T seconds
    T_switch = 2.0
    
    def time_switch(x, t, mode):
        # Switch at multiples of T_switch
        return (t % (3 * T_switch)) >= (mode + 1) * T_switch
    
    sim = SwitchedSystemSimulator(
        A=[A1, A2, A3],
        switching_conditions=[time_switch, time_switch, time_switch],
        mode_names=['Slow Decay', 'Medium Decay', 'Fast Decay']
    )
    
    x0 = np.array([5.0, 3.0])
    result = sim.simulate(x0, t_span=(0, 20), dt=0.05)
    
    print(f"Status: {result['message']}")
    print(f"Switches: {result['n_switches']}")
    print(f"Switch times: {[f'{t:.2f}' for t in result['switch_times'][:10]]}")
    
    sim.plot_trajectory(result, save_path='../images/example2_trajectory.png', show=False)
    print("Plot saved!")


def example_3_autonomous_system():
    """Example 3: Autonomous switched system (no control)."""
    print("\n" + "="*70)
    print("Example 3: Autonomous Switched System")
    print("="*70)
    
    # Van der Pol-like modes
    A1 = np.array([[0, 1], [-1, 0.5]])   # Limit cycle mode
    A2 = np.array([[0, 1], [-4, -2]])    # Stable focus
    
    # Switch based on distance from origin
    def radial_switch(x, t, mode):
        r = np.linalg.norm(x)
        if mode == 0:
            return r > 3.0
        else:
            return r < 1.5
    
    sim = SwitchedSystemSimulator(
        A=[A1, A2],
        B=None,  # Autonomous
        switching_conditions=[radial_switch, radial_switch],
        mode_names=['Oscillatory', 'Contracting']
    )
    
    x0 = np.array([4.0, 0.0])
    result = sim.simulate(x0, t_span=(0, 50), dt=0.1)
    
    print(f"Status: {result['message']}")
    print(f"Switches: {result['n_switches']}")
    
    sim.plot_trajectory(result, save_path='../images/example3_trajectory.png', show=False)
    sim.plot_phase_portrait(result, save_path='../images/example3_phase.png', show=False)
    print("Plots saved!")


def example_4_custom_switching_map():
    """Example 4: Custom switching logic with switching map."""
    print("\n" + "="*70)
    print("Example 4: Custom Switching Map")
    print("="*70)
    
    # Four quadrant modes
    A = [
        np.array([[-1, 0.5], [-0.5, -1]]),   # Quadrant 1
        np.array([[-0.8, -0.3], [0.3, -0.8]]), # Quadrant 2
        np.array([[-1.2, 0.2], [-0.2, -1.2]]), # Quadrant 3
        np.array([[-0.9, 0.4], [-0.4, -0.9]])  # Quadrant 4
    ]
    
    # Switch when crossing axes
    def axis_crossing(x, t, mode):
        # Check if we crossed an axis
        return abs(x[0]) < 0.1 or abs(x[1]) < 0.1
    
    # Determine next mode based on quadrant
    def quadrant_map(x, t, current_mode):
        if x[0] >= 0 and x[1] >= 0:
            return 0
        elif x[0] < 0 and x[1] >= 0:
            return 1
        elif x[0] < 0 and x[1] < 0:
            return 2
        else:
            return 3
    
    sim = SwitchedSystemSimulator(
        A=A,
        switching_conditions=[axis_crossing] * 4,
        switching_map=quadrant_map,
        mode_names=['Q1', 'Q2', 'Q3', 'Q4']
    )
    
    x0 = np.array([2.0, 0.1])
    result = sim.simulate(x0, t_span=(0, 40), dt=0.05)
    
    print(f"Status: {result['message']}")
    print(f"Switches: {result['n_switches']}")
    print(f"Mode sequence: {result['switch_modes'][:20]}")
    
    sim.plot_trajectory(result, save_path='../images/example4_trajectory.png', show=False)
    sim.plot_phase_portrait(result, save_path='../images/example4_phase.png', show=False)
    print("Plots saved!")


def example_5_nonlinear_system():
    """Example 5: Switched system with nonlinear terms."""
    print("\n" + "="*70)
    print("Example 5: Switched System with Nonlinear Dynamics")
    print("="*70)
    
    # Linear parts
    A1 = np.array([[0, 1], [-1, -0.2]])
    A2 = np.array([[0, 1], [-4, -1]])
    
    # Nonlinear terms
    def nonlinear1(x, u, t):
        # Cubic damping
        return np.array([0, -0.1 * x[1]**3])
    
    def nonlinear2(x, u, t):
        # Van der Pol-like term
        return np.array([0, (1 - x[0]**2) * x[1]])
    
    # Energy-based switching
    def energy_switch(x, t, mode):
        energy = 0.5 * (x[0]**2 + x[1]**2)
        if mode == 0:
            return energy > 5.0
        else:
            return energy < 1.0
    
    sim = SwitchedSystemSimulator(
        A=[A1, A2],
        f=[nonlinear1, nonlinear2],
        switching_conditions=[energy_switch, energy_switch],
        mode_names=['Cubic Damping', 'Van der Pol']
    )
    
    x0 = np.array([3.0, 0.0])
    result = sim.simulate(x0, t_span=(0, 50), dt=0.05)
    
    print(f"Status: {result['message']}")
    print(f"Switches: {result['n_switches']}")
    
    sim.plot_trajectory(result, save_path='../images/example5_trajectory.png', show=False)
    sim.plot_phase_portrait(result, save_path='../images/example5_phase.png', show=False)
    print("Plots saved!")


if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    # Run examples
    example_1_state_based_switching()
    example_2_time_based_switching()
    example_3_autonomous_system()
    example_4_custom_switching_map()
    example_5_nonlinear_system()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
