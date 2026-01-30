"""
Visualization tools for fluid dynamics simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, Optional


def plot_surface_evolution(solver, times_to_plot=None, ax=None):
    """
    Plot surface elevation at multiple times
    
    Args:
        solver: FluidSolver with history
        times_to_plot: Indices to plot (default: 5 evenly spaced)
        ax: Matplotlib axis (creates new if None)
    
    Returns:
        axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    history = solver.history
    x_vec = np.linspace(0, solver.grid['L'], solver.nx, endpoint=False)
    
    if times_to_plot is None:
        n_plots = min(5, len(history['eta']))
        times_to_plot = np.linspace(0, len(history['eta'])-1, n_plots, dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
    
    for idx, color in zip(times_to_plot, colors):
        ax.plot(x_vec, history['eta'][idx], 
               label=f't={history["time"][idx]:.2f}s',
               linewidth=2, color=color)
    
    ax.set_xlabel('x (dimensionless)')
    ax.set_ylabel('Surface elevation η')
    ax.set_title('Surface Elevation Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    return ax


def plot_velocity_field(solver, skip=None, ax=None):
    """
    Plot velocity field with quiver
    
    Args:
        solver: FluidSolver
        skip: Skip factor for arrows (default: auto)
        ax: Matplotlib axis
    
    Returns:
        axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    u_x, u_z = solver.compute_velocity_field()
    
    x_vec = np.linspace(0, solver.grid['L'], solver.nx, endpoint=False)
    z_vec = np.linspace(0, solver.grid['D'], solver.nz)
    X, Z = np.meshgrid(x_vec, z_vec)
    
    if skip is None:
        skip = max(1, solver.nx // 20)
    
    # Background: potential field
    ax.contourf(X, Z, solver.state.phi, levels=15, cmap='RdBu_r', alpha=0.5)
    
    # Arrows
    ax.quiver(X[::2, ::skip], Z[::2, ::skip],
             u_x[::2, ::skip], u_z[::2, ::skip],
             scale=10, alpha=0.8, width=0.003)
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('Velocity Field u = ∇φ + w')
    ax.set_aspect('equal')
    
    return ax


def plot_phase_space(solver, ax=None):
    """
    Plot phase space trajectory (η vs dη/dt at center)
    
    Args:
        solver: FluidSolver with history
        ax: Matplotlib axis
    
    Returns:
        axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    history = solver.history
    eta_center = [e[solver.nx // 2] for e in history['eta']]
    
    if len(history['time']) > 1:
        dt_save = history['time'][1] - history['time'][0]
        deta_center = np.gradient(eta_center, dt_save)
    else:
        deta_center = np.zeros_like(eta_center)
    
    ax.plot(eta_center, deta_center, linewidth=1, alpha=0.7)
    ax.scatter(eta_center[0], deta_center[0], c='green', s=100, 
              label='Start', zorder=5)
    ax.scatter(eta_center[-1], deta_center[-1], c='red', s=100, 
              label='End', zorder=5)
    
    ax.set_xlabel('η (center)')
    ax.set_ylabel('dη/dt (center)')
    ax.set_title('Phase Space Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_energy(solver, ax=None):
    """
    Plot energy components over time
    
    Args:
        solver: FluidSolver with history
        ax: Matplotlib axis
    
    Returns:
        axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    history = solver.history
    dx = solver.grid['dx']
    Fr, We = solver.dims['Fr'], solver.dims['We']
    
    # Compute energies
    surface_energy = [(1/We) * np.sum(e**2) * dx for e in history['eta']]
    potential_energy = [(1/Fr) * np.sum(np.abs(e)) * dx for e in history['eta']]
    
    ax.plot(history['time'], surface_energy, 
           label='Surface ∝ ∫η² dx', linewidth=2)
    ax.plot(history['time'], potential_energy, 
           label='Potential ∝ ∫|η| dx', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_diagnostics(solver, figsize=(16, 10), save_path=None):
    """
    Create comprehensive diagnostic plot
    
    Args:
        solver: FluidSolver with history
        figsize: Figure size
        save_path: Path to save figure (optional)
    
    Returns:
        figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Surface evolution
    ax1 = fig.add_subplot(3, 3, 1)
    plot_surface_evolution(solver, ax=ax1)
    
    # 2. Final potential field
    ax2 = fig.add_subplot(3, 3, 2)
    x_vec = np.linspace(0, solver.grid['L'], solver.nx, endpoint=False)
    z_vec = np.linspace(0, solver.grid['D'], solver.nz)
    X, Z = np.meshgrid(x_vec, z_vec)
    c = ax2.contourf(X, Z, solver.state.phi, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('Final Velocity Potential φ')
    ax2.set_aspect('equal')
    plt.colorbar(c, ax=ax2)
    
    # 3. Velocity field
    ax3 = fig.add_subplot(3, 3, 3)
    plot_velocity_field(solver, ax=ax3)
    
    # 4. Surface overlaid
    ax4 = fig.add_subplot(3, 3, 4)
    history = solver.history
    for idx in range(0, len(history['eta']), max(1, len(history['eta'])//10)):
        alpha = 0.3 + 0.7 * idx / len(history['eta'])
        ax4.plot(x_vec, history['eta'][idx], alpha=alpha, 
                color='blue', linewidth=1)
    ax4.plot(x_vec, history['eta'][-1], color='red', 
            linewidth=2, label='Final')
    ax4.set_xlabel('x')
    ax4.set_ylabel('η')
    ax4.set_title('Surface Evolution (overlaid)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy
    ax5 = fig.add_subplot(3, 3, 5)
    plot_energy(solver, ax=ax5)
    
    # 6. w³ at surface
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(x_vec, solver.state.w3[-1, :], linewidth=2)
    ax6.set_xlabel('x')
    ax6.set_ylabel('w³')
    ax6.set_title('Final w³ at Surface')
    ax6.grid(True, alpha=0.3)
    
    # 7. Vertical velocity profile
    ax7 = fig.add_subplot(3, 3, 7)
    u_x, u_z = solver.compute_velocity_field()
    j_center = solver.nx // 2
    ax7.plot(u_z[:, j_center], z_vec, linewidth=2)
    ax7.set_ylabel('z')
    ax7.set_xlabel('Vertical velocity u_z')
    ax7.set_title(f'Velocity Profile at Center')
    ax7.grid(True, alpha=0.3)
    
    # 8. Surface curvature
    ax8 = fig.add_subplot(3, 3, 8)
    from .operators import compute_curvature
    kappa = compute_curvature(solver.state.eta, solver.grid['dx'], "cpu")
    ax8.plot(x_vec, kappa, linewidth=2)
    ax8.set_xlabel('x')
    ax8.set_ylabel('κ[η]')
    ax8.set_title('Final Surface Curvature')
    ax8.grid(True, alpha=0.3)
    
    # 9. Phase space
    ax9 = fig.add_subplot(3, 3, 9)
    plot_phase_space(solver, ax=ax9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")
    
    return fig


def create_animation(solver, fps=10, save_path=None):
    """
    Create animation of surface evolution
    
    Args:
        solver: FluidSolver with history
        fps: Frames per second
        save_path: Path to save animation (optional)
    
    Returns:
        FuncAnimation object
    """
    history = solver.history
    x_vec = np.linspace(0, solver.grid['L'], solver.nx, endpoint=False)
    z_vec = np.linspace(0, solver.grid['D'], solver.nz)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Surface
    line_surf, = axes[0].plot(x_vec, history['eta'][0], 'b-', linewidth=2)
    axes[0].fill_between(x_vec, 0, history['eta'][0], alpha=0.3)
    axes[0].set_xlabel('x (dimensionless)')
    axes[0].set_ylabel('η')
    
    eta_min = min(np.min(e) for e in history['eta'])
    eta_max = max(np.max(e) for e in history['eta'])
    margin = max(0.01, (eta_max - eta_min) * 0.2)
    axes[0].set_ylim([eta_min - margin, eta_max + margin])
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    title_surf = axes[0].set_title(f'Surface Elevation - t = {history["time"][0]:.3f}s')
    
    # Potential
    vmin = min(np.min(p) for p in history['phi'])
    vmax = max(np.max(p) for p in history['phi'])
    if abs(vmax - vmin) < 1e-10:
        vmin, vmax = -0.01, 0.01
    
    im_phi = axes[1].imshow(history['phi'][0], origin='lower', 
                           extent=[0, solver.grid['L'], 0, solver.grid['D']],
                           aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[1].set_title('Velocity Potential φ')
    plt.colorbar(im_phi, ax=axes[1])
    
    def animate(frame):
        line_surf.set_ydata(history['eta'][frame])
        
        for coll in axes[0].collections:
            coll.remove()
        axes[0].fill_between(x_vec, 0, history['eta'][frame], alpha=0.3)
        
        im_phi.set_data(history['phi'][frame])
        title_surf.set_text(f'Surface Elevation - t = {history["time"][frame]:.3f}s')
        
        return line_surf, im_phi
    
    anim = FuncAnimation(fig, animate, frames=len(history['eta']),
                        interval=1000/fps, blit=False, repeat=True)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print(f"Saved animation to {save_path}")
    
    return anim


def plot_parameter_sweep(results, param_name, metric='max_eta', ax=None):
    """
    Plot results from parameter sweep
    
    Args:
        results: List of (param_value, final_state, solver) from run_parameter_sweep
        param_name: Name of varied parameter
        metric: Metric to plot ('max_eta', 'final_eta_center', 'energy_decay')
        ax: Matplotlib axis
    
    Returns:
        axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    param_values = [r[0] for r in results]
    
    if metric == 'max_eta':
        values = [np.max(np.abs(r[1].eta)) for r in results]
        ylabel = 'max|η|'
    elif metric == 'final_eta_center':
        values = [r[1].eta[len(r[1].eta)//2] for r in results]
        ylabel = 'η (center)'
    elif metric == 'energy_decay':
        values = []
        for _, state, solver in results:
            E0 = np.sum(solver.history['eta'][0]**2)
            Ef = np.sum(solver.history['eta'][-1]**2)
            values.append(1 - Ef/E0 if E0 > 0 else 0)
        ylabel = 'Energy decay fraction'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    ax.plot(param_values, values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs {param_name}')
    ax.grid(True, alpha=0.3)
    
    return ax


if __name__ == "__main__":
    from .config import get_preset
    from .solver import FluidSolver
    
    # Test visualization
    config = get_preset("fast")
    solver = FluidSolver(config)
    solver.run()
    
    # Create plots
    fig = plot_diagnostics(solver)
    plt.show()
